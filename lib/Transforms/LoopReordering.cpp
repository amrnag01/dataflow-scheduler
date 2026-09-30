//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on the "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Pass: -loop-reordering
//
// Identifies groups of scf.for loops that are candidates for interchange,
// clears the interleaved ops between them (hoisting or sinking as appropriate),
// and returns the resulting perfectly-nested candidate sets.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/Hoisting.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "loop-reordering"
#define DEBUG_LOOPORDER

using namespace mlir;

namespace scheduler {
#define GEN_PASS_DEF_LOOPREORDERINGPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

enum class OpClass {
  Hoistable,             ///< must be moved above the outer loop
  Sinkable,              ///< must be moved into the inner loop body
  HoistableAndSinkable,  ///< unconstrained; defaults to Sinkable
  Barrier,               ///< true barrier — cannot be moved
};

static StringRef opClassName(OpClass cls) {
  switch (cls) {
    case OpClass::Hoistable:
      return "hoistable";
    case OpClass::Sinkable:
      return "sinkable";
    case OpClass::HoistableAndSinkable:
      return "hoistable+sinkable";
    case OpClass::Barrier:
      return "TRUE BARRIER";
  }
  return "unknown";
}

static llvm::DenseMap<Operation*, OpClass> classifyInterleavedOps(
    scf::ForOp parentLoop, scf::ForOp childLoop) {
  // Step 1: Collect the set of operations that exist between the parentLoop and
  // the childLoop.
  SmallVector<Operation*> ops;
  for (Operation& op : *parentLoop.getBody()) {
    if (&op == childLoop.getOperation()) break;
    if (op.hasTrait<OpTrait::IsTerminator>()) continue;
    ops.push_back(&op);
  }
  llvm::DenseSet<Operation*> opSet(ops.begin(), ops.end());

  // Step 2: Mark each op as hoistable, sinkable, hoistable+sinkable(both), or
  // barrier
  Region& outerBody = parentLoop.getBodyRegion();
  Region& innerBody = childLoop.getBodyRegion();
  llvm::DenseMap<Operation*, OpClass> opCls;

  auto definedInParentLoopBody = [&](Value val) -> bool {
    if (auto blockArg = dyn_cast<BlockArgument>(val))
      return blockArg.getParentRegion() == &outerBody;
    return outerBody.isAncestor(val.getDefiningOp()->getParentRegion());
  };

  auto resolveOpCls = [&](Operation* op, OpClass target) {
    for (Value operand : op->getOperands()) {
      if (dyn_cast<BlockArgument>(operand)) continue;
      Operation* defOp = operand.getDefiningOp();
      if (opSet.count(defOp) && opCls[defOp] == OpClass::HoistableAndSinkable)
        opCls[defOp] = target;
    }
  };

  for (Operation* op : ops) {
    // Step 2a: Check if op is hoistable -- all operands outside parent loop or
    // from other hoistable/both ops in the intermediate op set.
    bool hoistable = true;
    for (Value operand : op->getOperands()) {
      if (!definedInParentLoopBody(operand)) continue;
      Operation* defOp = operand.getDefiningOp();
      if (!opSet.count(defOp) ||
          (opCls[defOp] != OpClass::Hoistable &&
           opCls[defOp] != OpClass::HoistableAndSinkable)) {
        hoistable = false;
        break;
      }
    }

    // Step 2b: Check if op is sinkable -- results consumed only inside child
    // loop body or in the set of intermediate ops
    bool sinkable = true;
    for (Value result : op->getResults()) {
      for (OpOperand& use : result.getUses()) {
        Operation* user = use.getOwner();
        if (innerBody.isAncestor(user->getParentRegion())) continue;
        if (opSet.count(user)) continue;
        sinkable = false;
        break;
      }
      if (!sinkable) break;
    }
    // Ensure that consumers of sinkable ops in the set of intermediate ops are
    // also sinkable
    if (sinkable) {
      for (Value operand : op->getOperands()) {
        if (!definedInParentLoopBody(operand)) continue;
        if (dyn_cast<BlockArgument>(operand)) continue;  // outer IV — ok
        Operation* defOp = operand.getDefiningOp();
        if (!opSet.count(defOp) ||
            (opCls[defOp] != OpClass::Sinkable &&
             opCls[defOp] != OpClass::HoistableAndSinkable)) {
          sinkable = false;
          break;
        }
      }
    }

    // Note: ops that are neither hoistable nor sinkable are barriers that
    // prevent loop interchanges between the parent and child loops
    OpClass c;
    if (hoistable && sinkable)
      c = OpClass::HoistableAndSinkable;
    else if (hoistable)
      c = OpClass::Hoistable;
    else if (sinkable)
      c = OpClass::Sinkable;
    else
      c = OpClass::Barrier;
    opCls[op] = c;

    // Step 2c: If the op is marked as hoistable, make sure that all its
    // consumer ops in the intermediate set are also marked as hoistable only.
    // If they were previously marked as hoistable+sinkable, they are now only
    // hoistable. Similarly, ensure consumers of sinkable ops in the
    // intermediate op set are marked as sinkable only.
    if (c == OpClass::Hoistable)
      resolveOpCls(op, OpClass::Hoistable);
    else if (c == OpClass::Sinkable)
      resolveOpCls(op, OpClass::Sinkable);
  }

  // Step 2d: For ops that are still marked as both hoistable and sinkable,
  // simply mark them as sinkable only (can be optimized later)
  for (Operation* op : ops)
    if (opCls[op] == OpClass::HoistableAndSinkable)
      opCls[op] = OpClass::Sinkable;

  return opCls;
}

struct InterleavedOpInfo {
  std::string text;
  OpClass cls;
};

static bool clearAndCheckCandidates(
    scf::ForOp parentLoop, scf::ForOp childLoop,
    SmallVectorImpl<InterleavedOpInfo>& opsOut) {
  // Step 1: Collect the set of operations between the parentLoop and the
  // childLoop, and classify them as hoistable, sinkable or barrier. If any
  // intermediate op is a barrier (cannot be hoisted above the parentLoop or
  // sunk into the childLoop's body), then the two loops are not candidates for
  // interchange.
  auto cls = classifyInterleavedOps(parentLoop, childLoop);
  for (auto& [op, c] : cls)
    if (c == OpClass::Barrier) return false;

  // Collect ops in original top-down order.
  SmallVector<Operation*> ops;
  for (Operation& op : *parentLoop.getBody()) {
    if (&op == childLoop.getOperation()) break;
    if (op.hasTrait<OpTrait::IsTerminator>()) continue;
    ops.push_back(&op);
  }

  // Capture op text + classification before moving (useful for debug prints).
  for (Operation* op : ops) {
    std::string buf;
    llvm::raw_string_ostream ss(buf);
    op->print(ss, OpPrintingFlags().useLocalScope());
    opsOut.push_back({std::move(buf), cls[op]});
  }

  // Step 2: Move hoistable ops as high as SSA allows.
  for (Operation* op : ops) {
    if (cls[op] != OpClass::Hoistable) continue;
    Operation* target = scheduler::findHoistingTarget(
        op, [](mlir::Region* r) { return isa<scf::ForOp>(r->getParentOp()); });
    if (target) op->moveBefore(target);
  }

  // Step 3: Move sinkable ops into childLoop body (reverse to preserve order).
  Block* innerBody = childLoop.getBody();
  for (Operation* op : llvm::reverse(ops))
    if (cls[op] == OpClass::Sinkable)
      op->moveBefore(innerBody, innerBody->begin());

  return true;
}

/// Returns the unique directly-nested scf.ForOp inside `loop`, or nullptr.
static scf::ForOp getUniqueInnerLoop(scf::ForOp loop) {
  scf::ForOp inner;
  for (Operation& op : *loop.getBody()) {
    if (auto candidate = dyn_cast<scf::ForOp>(&op)) {
      if (inner) return {};
      inner = candidate;
    }
  }
  return inner;
}

static void printLoopSummary(scf::ForOp loop, unsigned depth,
                             llvm::raw_ostream& os) {
  os << "  [depth " << depth << "]  scf.for ";
  loop.getInductionVar().printAsOperand(os, OpPrintingFlags());
  os << " = ";
  loop.getLowerBound().printAsOperand(os, OpPrintingFlags());
  os << " to ";
  loop.getUpperBound().printAsOperand(os, OpPrintingFlags());
  os << " step ";
  loop.getStep().printAsOperand(os, OpPrintingFlags());
  os << "\n";
}

/// pairOps[i] holds the interleaved ops between set[i] and set[i+1].
static void printCandidateSet(unsigned idx, ArrayRef<scf::ForOp> set,
                              ArrayRef<SmallVector<InterleavedOpInfo>> pairOps,
                              llvm::raw_ostream& os) {
  os << "\n┌─ Candidate Set " << idx << " (" << set.size() << " loops) ";
  os << "─────────────────────────────────\n";
  unsigned baseDepth = 0;
  for (Operation* p = set[0]->getParentOp(); p; p = p->getParentOp())
    if (isa<scf::ForOp>(p)) ++baseDepth;
  for (unsigned i = 0; i < set.size(); ++i) {
    printLoopSummary(set[i], baseDepth + i, os);
    if (i + 1 < set.size() && !pairOps[i].empty()) {
      for (auto& info : pairOps[i]) {
        os << "      [" << opClassName(info.cls) << "]  " << info.text << "\n";
      }
    }
  }
  os << "└─────────────────────────────────────────────────────────────\n";
}

struct CandidateSetInfo {
  SmallVector<scf::ForOp> loops;
  SmallVector<SmallVector<InterleavedOpInfo>>
      pairOps;  ///< pairOps[i]: between loops[i] and loops[i+1]
};

static void findCandidateSets(
    Operation* root, SmallVectorImpl<CandidateSetInfo>& candidateSets) {
  // Step 1: Find starting points (startLoops) for identifying candidate sets.
  // Every loop whose immediate parent is not another loop is a potential
  // starting point to explore, since loops perfectly nested with parent loops
  // will always simply be placed in the candidate set started by its parent
  // loop.
  SmallVector<scf::ForOp> start_loops;
  root->walk([&](scf::ForOp loop) {
    if (!isa<scf::ForOp>(loop->getParentOp())) start_loops.push_back(loop);
  });

  // Step 2: Identify candidate sets around each startLoop. All loops placed
  // inside a candidate set must be able to be ordered arbitrarily.
  for (scf::ForOp startLoop : start_loops) {
    CandidateSetInfo current_set;
    scf::ForOp current_loop = startLoop;

    while (current_loop) {
      if (current_set.loops.empty()) current_set.loops.push_back(current_loop);

      // 2a: Continue growing the set only if the current loop has a unique
      // child loop in its body. If multiple child loops are found, they cannot
      // be part of the same candidate set as the current loop, and so the set
      // is complete. If no child loop is found, the set is complete.
      scf::ForOp child_loop = getUniqueInnerLoop(current_loop);
      if (!child_loop) break;

      // 2b: Check if the current loop should be added to the current set. It
      // can be added if the set of operations between the current loop and its
      // parent loop is either empty, or fully hoistable and/or sinkable. If
      // hoisting or sinking is needed to add the loop to the current set, then
      // need to move these ops also.
      SmallVector<InterleavedOpInfo> pairInfo;
      if (clearAndCheckCandidates(current_loop, child_loop, pairInfo)) {
        current_set.loops.push_back(child_loop);
        current_set.pairOps.push_back(std::move(pairInfo));
        current_loop = child_loop;
      } else {
        if (current_set.loops.size() >= 2)
          candidateSets.push_back(std::move(current_set));
        current_set = {};
        current_set.loops.push_back(child_loop);
        current_loop = child_loop;
      }
    }

    // 2c: If the current set has more than 1 loop, reorderings are possible.
    if (current_set.loops.size() >= 2)
      candidateSets.push_back(std::move(current_set));
  }
}

static void interchangeLoops(scf::ForOp forOpA, scf::ForOp forOpB) {
  assert(&*forOpA.getBody()->begin() == forOpB.getOperation());
  auto& forOpABody = forOpA.getBody()->getOperations();
  auto& forOpBBody = forOpB.getBody()->getOperations();

  // Step 1: Splice forOpB out of forOpA's body to just before forOpA in the
  // parent block. Now, forOpA's body now contains only its terminator.
  forOpA->getBlock()->getOperations().splice(Block::iterator(forOpA),
                                             forOpABody, forOpABody.begin(),
                                             std::prev(forOpABody.end()));

  // Step 2: Splice the original forOpB body contents into the beginning of
  // forOpA's empty body. Now, forOpA's body contains the forOpB's original body
  forOpABody.splice(forOpABody.begin(), forOpBBody, forOpBBody.begin(),
                    std::prev(forOpBBody.end()));

  // Step 3: Splice forOpA into the beginning of forOpB's body so forOpB wraps
  // forOpA. Now, forOpB's body contains forOpA + forOpB's original body
  // (interchange complete)
  forOpBBody.splice(forOpBBody.begin(), forOpA->getBlock()->getOperations(),
                    Block::iterator(forOpA));
}

[[maybe_unused]] static void reorderLoops(ArrayRef<scf::ForOp> loops,
                                          ArrayRef<unsigned> permutation) {
  assert(loops.size() == permutation.size());

  // The helper interchangeLoops() can interchange pairs of scf.for loops,
  // similar to mlir::affine::interchangeLoops. To arrange the given loops in
  // the specified permutation order, need to bubble sort loops into their
  // intended position through pairwise exchanges.
  SmallVector<scf::ForOp> current_loops(loops.begin(), loops.end());
  SmallVector<unsigned> pos(loops.size());
  for (unsigned i = 0; i < loops.size(); ++i) pos[i] = i;

  // For each target position i (outermost first), bubble the desired loop
  // down from its current position using adjacent interchanges.
  for (unsigned i = 0; i < permutation.size(); ++i) {
    unsigned orig_pos_of_wanted_loop =
        permutation[i];  // original index we want at position i
    unsigned curr_pos_of_wanted_loop =
        pos[orig_pos_of_wanted_loop];  // where it currently sits

    // Bubble it up (swap with predecessor) until it reaches position wanted.
    while (curr_pos_of_wanted_loop > i) {
      interchangeLoops(current_loops[curr_pos_of_wanted_loop - 1],
                       current_loops[curr_pos_of_wanted_loop]);
      // Update tracking.
      unsigned orig_above = 0;
      for (unsigned j = 0; j < loops.size(); ++j)
        if (pos[j] == curr_pos_of_wanted_loop - 1) {
          orig_above = j;
          break;
        }
      std::swap(current_loops[curr_pos_of_wanted_loop - 1],
                current_loops[curr_pos_of_wanted_loop]);
      pos[orig_above] = curr_pos_of_wanted_loop;
      pos[orig_pos_of_wanted_loop] = curr_pos_of_wanted_loop - 1;
      --curr_pos_of_wanted_loop;
    }
  }
}

// ---------------------------------------------------------------------------

struct LoopReorderingPass
    : public scheduler::impl::LoopReorderingPassBase<LoopReorderingPass> {
  using LoopReorderingPassBase<LoopReorderingPass>::LoopReorderingPassBase;

  void runOnOperation() override {
    Operation* module = getOperation();

    // Step 1: Identify candidate sets of loops for reordering. All loops that
    // fall into a set can be ordered arbitrarily. If ops in between loops
    // create false barriers, they are hoisted/sunk to allow reordering
    SmallVector<CandidateSetInfo> candidateSets;
    findCandidateSets(module, candidateSets);

#ifdef DEBUG_LOOPORDER
    if (candidateSets.empty()) {
      llvm::outs() << "[LoopReordering] No interchange-candidate loop sets "
                      "found.\n";
      return;
    }

    llvm::outs() << "[LoopReordering] Found " << candidateSets.size()
                 << " interchange-candidate loop set(s):";
    for (auto [idx, info] : llvm::enumerate(candidateSets))
      printCandidateSet(static_cast<unsigned>(idx), info.loops, info.pairOps,
                        llvm::outs());
    llvm::outs() << "\n";
#endif

    // Step 2(temp): Test reorderLoops by reversing every candidate set.
    /*for (auto& info : candidateSets) {
      unsigned n = info.loops.size();
      SmallVector<unsigned> perm(n);
      for (unsigned i = 0; i < n; ++i) perm[i] = n - 1 - i;
      reorderLoops(info.loops, perm);
    }*/
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createLoopReorderingPass() {
  return std::make_unique<LoopReorderingPass>();
}
