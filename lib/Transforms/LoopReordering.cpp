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
// Identifies groups of scf.for loops that are candidates for interchange.
// Two adjacent loops are candidates if there is no "true barrier" op between
// them — i.e. every op interleaved between the two loop headers is either
// hoistable (all operands defined outside the outer loop) or sinkable (result
// only used inside the inner loop body). Candidacy is extended transitively to
// form maximal interchange-candidate sets.
//
// Currently this pass only prints the candidate sets and makes no IR changes.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define PASS_NAME "loop-reordering"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace scheduler {
#define GEN_PASS_DEF_LOOPREORDERINGPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Classification of a single interleaved op.
/// An op may be both Hoistable and Sinkable simultaneously.
struct OpClass {
  bool hoistable = false;
  bool sinkable  = false;
  bool isTrueBarrier() const { return !hoistable && !sinkable; }
};

static std::string opClassName(OpClass cls) {
  if (cls.hoistable && cls.sinkable) return "hoistable+sinkable";
  if (cls.hoistable) return "hoistable";
  if (cls.sinkable)  return "sinkable";
  return "TRUE BARRIER";
}

/// Classify all ops between `outerLoop` and `innerLoop` in a single top-down
/// pass. An op can be both hoistable and sinkable.
///
/// Hoistable: all operands are defined outside the outer loop, or produced by
///   a prior op in S that is hoistable.
///
/// Sinkable: all results are used only inside the inner loop body, on the
///   inner loop op itself, or by a later op in S — AND all operands are either
///   outside the outer loop, produced by a hoistable op in S, or produced by
///   a sinkable op in S (so the whole chain can move down together).
static llvm::DenseMap<Operation*, OpClass> classifyInterleavedOps(
    scf::ForOp outerLoop, scf::ForOp innerLoop) {
  // Collect interleaved ops in order.
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

  // Single top-down pass.
  for (Operation* op : ops) {
    OpClass c;

    // --- Hoistable check ---
    // All operands must be outside the outer loop or from a hoistable op in S.
    c.hoistable = true;
    for (Value operand : op->getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (blockArg.getParentRegion() == &outerBody) {
          c.hoistable = false;
          break;
        }
      } else {
        Operation* defOp = operand.getDefiningOp();
        if (outerBody.isAncestor(defOp->getParentRegion())) {
          // Defined inside the outer loop — only ok if it's a hoistable op in S.
          if (!opSet.count(defOp) || !cls[defOp].hoistable) {
            c.hoistable = false;
            break;
          }
        }
      }
    }

    // --- Sinkable check ---
    // (1) All results only consumed inside inner loop body, by inner loop
    //     itself, or by a later op in S.
    // (2) All operands are outside the outer loop, from a hoistable op in S,
    //     or from a sinkable op in S (so the chain can travel down together).
    c.sinkable = true;

    // Check (1): result uses.
    for (Value result : op->getResults()) {
      for (OpOperand& use : result.getUses()) {
        Operation* user = use.getOwner();
        if (user == innerLoop.getOperation()) continue;
        if (innerBody.isAncestor(user->getParentRegion())) continue;
        if (opSet.count(user)) continue;  // later op in S — ok
        c.sinkable = false;
        break;
      }
      if (!c.sinkable) break;
    }

    // Check (2): operand provenance (only if (1) passed).
    if (c.sinkable) {
      for (Value operand : op->getOperands()) {
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (blockArg.getParentRegion() == &outerBody) {
            // Outer IV or iter-arg — fine, the sink will carry it along.
            continue;
          }
        } else {
          Operation* defOp = operand.getDefiningOp();
          if (outerBody.isAncestor(defOp->getParentRegion())) {
            // Defined inside outer loop: must be hoistable or sinkable in S.
            if (!opSet.count(defOp) ||
                (!cls[defOp].hoistable && !cls[defOp].sinkable)) {
              c.sinkable = false;
              break;
            }
          }
        }
      }
    }

    cls[op] = c;
  }

  return cls;
}

/// Returns true if there is no true-barrier op between `outerLoop` and
/// `innerLoop` (i.e. the pair is interchange-candidate).
static bool areCandidates(scf::ForOp outerLoop, scf::ForOp innerLoop) {
  auto cls = classifyInterleavedOps(outerLoop, innerLoop);
  for (auto& [op, c] : cls)
    if (c.isTrueBarrier()) return false;
  return true;
}

/// Returns the unique directly-nested scf.ForOp inside `loop`, if there is
/// exactly one and it appears as the *only* structural op in the body (all
/// other ops are either hoistable or sinkable w.r.t. that pair).  Returns
/// nullptr if there is no directly nested loop, or if there are multiple.
static scf::ForOp getUniqueInnerLoop(scf::ForOp loop) {
  scf::ForOp inner;
  for (Operation& op : *loop.getBody()) {
    if (auto candidate = dyn_cast<scf::ForOp>(&op)) {
      if (inner)
        return {};  // more than one inner loop — don't treat as a nest
      inner = candidate;
    }
  }
  return inner;
}

/// Walk the IR and find all maximal interchange-candidate loop sets.
///
/// Algorithm:
///   1. Find all top-level scf.for loops (not nested inside another scf.for).
///   2. For each top-level loop, descend the nesting chain.  At each level,
///      check whether the current loop and its unique inner loop are candidates.
///   3. Extend the current candidate set transitively as long as the pair
///      passes the areCandidates() check.
///   4. Emit any set with >= 2 members.
static void findCandidateSets(
    Operation* root,
    SmallVectorImpl<SmallVector<scf::ForOp>>& result) {

  // Collect all scf.for ops whose immediate parent is NOT another scf.for.
  SmallVector<scf::ForOp> topLevel;
  root->walk([&](scf::ForOp loop) {
    if (!isa<scf::ForOp>(loop->getParentOp()))
      topLevel.push_back(loop);
  });

  for (scf::ForOp startLoop : topLevel) {
    SmallVector<scf::ForOp> currentSet;
    scf::ForOp cur = startLoop;

    while (cur) {
      if (currentSet.empty()) {
        currentSet.push_back(cur);
      }

      scf::ForOp inner = getUniqueInnerLoop(cur);
      if (!inner)
        break;

      if (areCandidates(cur, inner)) {
        currentSet.push_back(inner);
        cur = inner;
      } else {
        // True barrier found — flush the current set if it has >= 2 loops,
        // then start a new set beginning with the inner loop.
        if (currentSet.size() >= 2)
          result.push_back(currentSet);
        currentSet.clear();
        currentSet.push_back(inner);
        cur = inner;
      }
    }

    if (currentSet.size() >= 2)
      result.push_back(currentSet);
  }
}

// ---------------------------------------------------------------------------
// Pretty-printing helpers
// ---------------------------------------------------------------------------

/// Print one scf.for loop's summary line, e.g.:
///   [depth 2]  scf.for %arg1 = %c0 to %7 step %c1
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

/// Print the interleaved ops between outerLoop and innerLoop with their
/// classifications (hoistable / sinkable / true barrier).
static void printInterleavedOps(scf::ForOp outerLoop, scf::ForOp innerLoop,
                                 llvm::raw_ostream& os) {
  auto cls = classifyInterleavedOps(outerLoop, innerLoop);
  if (cls.empty()) {
    os << "          (no interleaved ops — perfectly nested)\n";
    return;
  }
  // Print in original order.
  for (Operation& op : *outerLoop.getBody()) {
    if (&op == innerLoop.getOperation()) break;
    if (op.hasTrait<OpTrait::IsTerminator>()) continue;
    os << "          [" << opClassName(cls[&op]) << "]  ";
    op.print(os, OpPrintingFlags().useLocalScope());
    os << "\n";
  }
}

/// Pretty-print a single interchange-candidate set.
static void printCandidateSet(unsigned idx,
                               ArrayRef<scf::ForOp> set,
                               llvm::raw_ostream& os) {
  os << "\n┌─ Candidate Set " << idx << " (" << set.size() << " loops) ";
  os << "─────────────────────────────────\n";

  // Compute depth of the first loop in the set relative to any scf.for parent.
  unsigned baseDepth = 0;
  for (Operation* p = set[0]->getParentOp(); p; p = p->getParentOp())
    if (isa<scf::ForOp>(p)) ++baseDepth;

  for (unsigned i = 0; i < set.size(); ++i) {
    printLoopSummary(set[i], baseDepth + i, os);
    if (i + 1 < set.size()) {
      os << "        interleaved ops between loop " << i << " and loop "
         << (i + 1) << ":\n";
      printInterleavedOps(set[i], set[i + 1], os);
    }
  }
  os << "└─────────────────────────────────────────────────────────────\n";
}

// ---------------------------------------------------------------------------

struct LoopReorderingPass
    : public scheduler::impl::LoopReorderingPassBase<LoopReorderingPass> {
  using LoopReorderingPassBase<LoopReorderingPass>::LoopReorderingPassBase;

  void runOnOperation() override {
    Operation* module = getOperation();

    SmallVector<SmallVector<scf::ForOp>> candidateSets;
    findCandidateSets(module, candidateSets);

    if (candidateSets.empty()) {
      llvm::outs() << "[LoopReordering] No interchange-candidate loop sets "
                      "found.\n";
      return;
    }

    llvm::outs() << "[LoopReordering] Found " << candidateSets.size()
                 << " interchange-candidate loop set(s):";
    for (auto [idx, set] : llvm::enumerate(candidateSets))
      printCandidateSet(static_cast<unsigned>(idx), set, llvm::outs());
    llvm::outs() << "\n";

    // Dump the IR of each unique enclosing function so the reader can see the
    // full context of every candidate set.
    LLVM_DEBUG({
      // Collect unique enclosing func::FuncOp for all sets.
      llvm::SmallPtrSet<Operation*, 4> printed;
      for (auto& set : candidateSets) {
        // Walk up from the first loop in the set to find its enclosing FuncOp.
        Operation* parent = set[0]->getParentOp();
        while (parent && !isa<func::FuncOp>(parent))
          parent = parent->getParentOp();
        if (!parent || !printed.insert(parent).second) continue;

        llvm::dbgs() << "\n[LoopReordering] IR of enclosing function after "
                        "candidate analysis:\n";
        llvm::dbgs() << "// func: ";
        llvm::dbgs() << cast<func::FuncOp>(parent).getName() << "\n";
        parent->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n";
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createLoopReorderingPass() {
  return std::make_unique<LoopReorderingPass>();
}
