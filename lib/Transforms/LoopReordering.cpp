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
// distributed under the License is distributed on an "AS IS" BASIS,
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#define PASS_NAME "loop-reordering"
#define DEBUG_LOOPORDER

using namespace mlir;

namespace scheduler {
#define GEN_PASS_DEF_LOOPREORDERINGPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Classification of a single interleaved op.
enum class OpClass {
  Hoistable,            ///< must be moved above the outer loop
  Sinkable,             ///< must be moved into the inner loop body
  HoistableAndSinkable, ///< unconstrained; defaults to Sinkable
  Barrier,              ///< true barrier — cannot be moved
};

static StringRef opClassName(OpClass cls) {
  switch (cls) {
    case OpClass::Hoistable:            return "hoistable";
    case OpClass::Sinkable:             return "sinkable";
    case OpClass::HoistableAndSinkable: return "hoistable+sinkable";
    case OpClass::Barrier:              return "TRUE BARRIER";
  }
  return "unknown";
}

/// Classify all ops between `outerLoop` and `innerLoop` in a single top-down
/// pass with immediate demotion of HoistableAndSinkable producers.
static llvm::DenseMap<Operation*, OpClass> classifyInterleavedOps(
    scf::ForOp outerLoop, scf::ForOp innerLoop) {
  SmallVector<Operation*> ops;
  for (Operation& op : *outerLoop.getBody()) {
    if (&op == innerLoop.getOperation()) break;
    if (op.hasTrait<OpTrait::IsTerminator>()) continue;
    ops.push_back(&op);
  }

  llvm::DenseSet<Operation*> opSet(ops.begin(), ops.end());
  Region& outerBody = outerLoop.getBodyRegion();
  Region& innerBody = innerLoop.getBodyRegion();

  llvm::DenseMap<Operation*, OpClass> cls;

  auto definedInOuter = [&](Value val) -> bool {
    if (auto blockArg = dyn_cast<BlockArgument>(val))
      return blockArg.getParentRegion() == &outerBody;
    return outerBody.isAncestor(val.getDefiningOp()->getParentRegion());
  };

  auto demoteOperandsInS = [&](Operation* op, OpClass target) {
    for (Value operand : op->getOperands()) {
      if (dyn_cast<BlockArgument>(operand)) continue;
      Operation* defOp = operand.getDefiningOp();
      if (opSet.count(defOp) && cls[defOp] == OpClass::HoistableAndSinkable)
        cls[defOp] = target;
    }
  };

  for (Operation* op : ops) {
    // Hoistable: all operands outside outer loop or from Hoistable/Both in S.
    bool hoistable = true;
    for (Value operand : op->getOperands()) {
      if (!definedInOuter(operand)) continue;
      Operation* defOp = operand.getDefiningOp();
      if (!opSet.count(defOp) ||
          (cls[defOp] != OpClass::Hoistable &&
           cls[defOp] != OpClass::HoistableAndSinkable)) {
        hoistable = false;
        break;
      }
    }

    // Sinkable check (1): results only consumed inside inner body or later in S.
    // Note: use as a loop bound/step does NOT qualify — bounds are evaluated
    // before the body, so such ops must be hoistable, not sinkable.
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
    // Sinkable check (2): inner-loop-defined operands from Sinkable/Both in S.
    if (sinkable) {
      for (Value operand : op->getOperands()) {
        if (!definedInOuter(operand)) continue;
        if (dyn_cast<BlockArgument>(operand)) continue;  // outer IV — ok
        Operation* defOp = operand.getDefiningOp();
        if (!opSet.count(defOp) ||
            (cls[defOp] != OpClass::Sinkable &&
             cls[defOp] != OpClass::HoistableAndSinkable)) {
          sinkable = false;
          break;
        }
      }
    }

    OpClass c;
    if (hoistable && sinkable)  c = OpClass::HoistableAndSinkable;
    else if (hoistable)         c = OpClass::Hoistable;
    else if (sinkable)          c = OpClass::Sinkable;
    else                        c = OpClass::Barrier;
    cls[op] = c;

    if (c == OpClass::Hoistable)
      demoteOperandsInS(op, OpClass::Hoistable);
    else if (c == OpClass::Sinkable)
      demoteOperandsInS(op, OpClass::Sinkable);
  }

  // Remaining HoistableAndSinkable defaults to Sinkable.
  for (Operation* op : ops)
    if (cls[op] == OpClass::HoistableAndSinkable)
      cls[op] = OpClass::Sinkable;

  return cls;
}

/// Per-pair interleaved op info: op text + classification, captured before
/// moving (moving invalidates SSA names).
struct InterleavedOpInfo {
  std::string text;
  OpClass cls;
};

/// Checks if (outerLoop, innerLoop) are interchange-candidates.
/// If so: captures interleaved op info into `opsOut`, then moves the ops
/// (hoistable above outerLoop, sinkable into innerLoop body).
/// Returns true if the pair is a candidate; false if a barrier was found.
static bool clearAndCheckCandidates(
    scf::ForOp outerLoop, scf::ForOp innerLoop,
    SmallVectorImpl<InterleavedOpInfo>& opsOut) {
  auto cls = classifyInterleavedOps(outerLoop, innerLoop);
  for (auto& [op, c] : cls)
    if (c == OpClass::Barrier) return false;

  // Collect ops in original top-down order.
  SmallVector<Operation*> ops;
  for (Operation& op : *outerLoop.getBody()) {
    if (&op == innerLoop.getOperation()) break;
    if (op.hasTrait<OpTrait::IsTerminator>()) continue;
    ops.push_back(&op);
  }

  // Capture op text + classification before moving (names change after move).
  for (Operation* op : ops) {
    std::string buf;
    llvm::raw_string_ostream ss(buf);
    op->print(ss, OpPrintingFlags().useLocalScope());
    opsOut.push_back({std::move(buf), cls[op]});
  }

  // Move hoistable ops as high as SSA allows.
  for (Operation* op : ops) {
    if (cls[op] != OpClass::Hoistable) continue;
    Operation* target = scheduler::findHoistingTarget(
        op, [](mlir::Region* r) { return isa<scf::ForOp>(r->getParentOp()); });
    if (target)
      op->moveBefore(target);
  }

  // Move sinkable ops into innerLoop body (reverse to preserve order).
  Block* innerBody = innerLoop.getBody();
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
static void printCandidateSet(
    unsigned idx, ArrayRef<scf::ForOp> set,
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
  SmallVector<SmallVector<InterleavedOpInfo>> pairOps; ///< pairOps[i]: between loops[i] and loops[i+1]
};

static void findCandidateSets(Operation* root,
                               SmallVectorImpl<CandidateSetInfo>& result) {
  SmallVector<scf::ForOp> topLevel;
  root->walk([&](scf::ForOp loop) {
    if (!isa<scf::ForOp>(loop->getParentOp()))
      topLevel.push_back(loop);
  });

  for (scf::ForOp startLoop : topLevel) {
    CandidateSetInfo current;
    scf::ForOp cur = startLoop;

    while (cur) {
      if (current.loops.empty())
        current.loops.push_back(cur);

      scf::ForOp inner = getUniqueInnerLoop(cur);
      if (!inner) break;

      SmallVector<InterleavedOpInfo> pairInfo;
      if (clearAndCheckCandidates(cur, inner, pairInfo)) {
        current.loops.push_back(inner);
        current.pairOps.push_back(std::move(pairInfo));
        cur = inner;
      } else {
        if (current.loops.size() >= 2)
          result.push_back(std::move(current));
        current = {};
        current.loops.push_back(inner);
        cur = inner;
      }
    }

    if (current.loops.size() >= 2)
      result.push_back(std::move(current));
  }
}

// ---------------------------------------------------------------------------

/// Interchange two perfectly-nested adjacent scf.for loops.
/// forOpA must be the outer loop and forOpB must be the unique loop
/// directly nested inside forOpA's body (same invariant as affine::interchangeLoops).
static void interchangeLoops(scf::ForOp forOpA, scf::ForOp forOpB) {
  assert(&*forOpA.getBody()->begin() == forOpB.getOperation());
  auto& forOpABody = forOpA.getBody()->getOperations();
  auto& forOpBBody = forOpB.getBody()->getOperations();

  // 1) Splice forOpB out of forOpA's body to just before forOpA in the parent
  //    block. forOpA's body now contains only its terminator.
  forOpA->getBlock()->getOperations().splice(
      Block::iterator(forOpA), forOpABody,
      forOpABody.begin(), std::prev(forOpABody.end()));

  // 2) Splice the original forOpA body contents (now forOpB's former body,
  //    i.e. the real loop body) into the beginning of forOpA's empty body.
  forOpABody.splice(forOpABody.begin(), forOpBBody,
                    forOpBBody.begin(), std::prev(forOpBBody.end()));

  // 3) Splice forOpA into the beginning of forOpB's body so forOpB wraps forOpA.
  forOpBBody.splice(forOpBBody.begin(),
                    forOpA->getBlock()->getOperations(),
                    Block::iterator(forOpA));
}

/// Reorder the loops in `loops` (a perfectly-nested candidate set) according
/// to `permutation`, where permutation[i] is the index in the original `loops`
/// array that should become position i after reordering.
///
/// Example: loops = {A, B, C}, permutation = {2, 0, 1}
///   → new nesting order: C (outer), A, B (inner)
///
/// The permutation is decomposed into adjacent transpositions (bubble-sort)
/// and each is applied via interchangeLoops().
[[maybe_unused]] static void reorderLoops(ArrayRef<scf::ForOp> loops,
                                          ArrayRef<unsigned> permutation) {
  assert(loops.size() == permutation.size());

  // Work on a mutable copy tracking the current position of each original loop.
  SmallVector<scf::ForOp> current(loops.begin(), loops.end());

  // Build the inverse: pos[i] = current index of original loop i.
  SmallVector<unsigned> pos(loops.size());
  for (unsigned i = 0; i < loops.size(); ++i)
    pos[i] = i;

  // For each target position i (outermost first), bubble the desired loop
  // down from its current position using adjacent interchanges.
  for (unsigned i = 0; i < permutation.size(); ++i) {
    unsigned want = permutation[i]; // original index we want at position i
    unsigned cur = pos[want];       // where it currently sits

    // Bubble it up (swap with predecessor) until it reaches position i.
    while (cur > i) {
      interchangeLoops(current[cur - 1], current[cur]);
      // Update tracking.
      unsigned orig_above = 0;
      for (unsigned j = 0; j < loops.size(); ++j)
        if (pos[j] == cur - 1) { orig_above = j; break; }
      std::swap(current[cur - 1], current[cur]);
      pos[orig_above] = cur;
      pos[want] = cur - 1;
      --cur;
    }
  }
}

// ---------------------------------------------------------------------------

struct LoopReorderingPass
    : public scheduler::impl::LoopReorderingPassBase<LoopReorderingPass> {
  using LoopReorderingPassBase<LoopReorderingPass>::LoopReorderingPassBase;

  void runOnOperation() override {
    Operation* module = getOperation();

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

    // TODO(temp): test reorderLoops by reversing every candidate set.
    for (auto& info : candidateSets) {
      unsigned n = info.loops.size();
      SmallVector<unsigned> perm(n);
      for (unsigned i = 0; i < n; ++i)
        perm[i] = n - 1 - i;
      reorderLoops(info.loops, perm);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createLoopReorderingPass() {
  return std::make_unique<LoopReorderingPass>();
}
