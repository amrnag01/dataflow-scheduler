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
// Pass: -symbolize-loop-order-and-data-transfer
//
// Identifies groups of scf.for loops that are candidates for interchange,
// clears the interleaved ops between them (hoisting or sinking as appropriate),
// makes the loop order of each resulting perfectly-nested candidate set
// symbolic, and makes the data transfers conditional on that loop order.
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/PipelineScope.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/Hoisting.h"
#include "dataflow-scheduler/Transforms/Utils/LoopOrderSymbols.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "symbolize-loop-order-and-data-transfer"
#define DEBUG_LOOP_TILING

using namespace mlir;

namespace scheduler {
#define GEN_PASS_DEF_SYMBOLIZELOOPORDERANDDATATRANSFERPASS
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

static scf::ForOp getUniqueInnerLoop(scf::ForOp loop) {
  // Returns the unique directly-nested scf.ForOp inside `loop`, or nullptr.
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

[[maybe_unused]] static void printCandidateSet(
    unsigned idx, ArrayRef<scf::ForOp> set,
    ArrayRef<SmallVector<InterleavedOpInfo>> pairOps, llvm::raw_ostream& os) {
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
      pairOps;  // pairOps[i]: ops that lie between between loops[i] and
                // loops[i+1]
};

static bool boundsDependOnSetIVs(scf::ForOp loop,
                                 ArrayRef<scf::ForOp> setLoops) {
  // Returns true if lower bound, upper bound or step of `loop` transitively
  // depends on the induction variable of any loop in `setLoops`.
  scf::ForOp outermostLoop = setLoops.front();
  Operation* outermost = outermostLoop.getOperation();
  llvm::DenseSet<Value> setIVs;
  for (scf::ForOp l : setLoops) setIVs.insert(l.getInductionVar());

  SmallVector<Value> worklist = {loop.getLowerBound(), loop.getUpperBound(),
                                 loop.getStep()};
  llvm::DenseSet<Operation*> visited;
  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    if (setIVs.count(val)) return true;
    Operation* defOp = val.getDefiningOp();
    // Block arguments other than set IVs, and values defined outside the
    // outermost loop of the set, cannot depend on the set's IVs.
    if (!defOp || !outermost->isProperAncestor(defOp)) continue;
    if (!visited.insert(defOp).second) continue;
    worklist.append(defOp->operand_begin(), defOp->operand_end());
  }
  return false;
}

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
      // need to move these ops also. Note: the child loop cannot join the set
      // if its bounds depend on the IV of any loop already in the set.
      SmallVector<InterleavedOpInfo> pairInfo;
      if (!boundsDependOnSetIVs(child_loop, current_set.loops) &&
          clearAndCheckCandidates(current_loop, child_loop, pairInfo)) {
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

// Metadata describing how a candidate set was symbolized.
struct SymbolizedSetInfo {
  SmallVector<scf::ForOp> loops;   // Candidate set of loops for reordering,
                                   // arranged outermost to innermost
  SmallVector<Value> positions;    // positions[d]: index of the original loop
                                   // that runs at depth d.
  SmallVector<Value> selectedIVs;  // selectedIVs[j]: value standing for the IV
                                   // of original loop j.
};

static SmallVector<SymbolizedSetInfo> symbolizeLoopOrdering(
    ArrayRef<CandidateSetInfo> candidateSets) {
  SmallVector<SymbolizedSetInfo> symbolizedSets;

  // Map from candidate set to its position variables
  llvm::DenseMap<const CandidateSetInfo*, SmallVector<Value>>
      candidateSetPositions;

  // Step 1: For each candidate set, gather metadata and assert assumptions
  for (auto [candIdx, candidateSet] : llvm::enumerate(candidateSets)) {
    const auto& loops = candidateSet.loops;
    unsigned numLoops = loops.size();

    // Verify all loops have lower bound 0 and step 1
    for (unsigned i = 0; i < numLoops; ++i) {
      scf::ForOp loop = loops[i];

      // Check lower bound is constant 0
      Operation* lbOp = loop.getLowerBound().getDefiningOp();
      auto lbConstOp = dyn_cast_or_null<arith::ConstantOp>(lbOp);
      assert(lbConstOp && "Loop lower bound must be defined by arith.constant");
      auto lbAttr = lbConstOp.getValue();
      assert(isa<IntegerAttr>(lbAttr) &&
             cast<IntegerAttr>(lbAttr).getValue().isZero() &&
             "Loop lower bound must be constant 0");

      // Check step is constant 1
      Operation* stepOp = loop.getStep().getDefiningOp();
      auto stepConstOp = dyn_cast_or_null<arith::ConstantOp>(stepOp);
      assert(stepConstOp && "Loop step must be defined by arith.constant");
      auto stepAttr = stepConstOp.getValue();
      assert(isa<IntegerAttr>(stepAttr) &&
             cast<IntegerAttr>(stepAttr).getValue() == 1 &&
             "Loop step must be constant 1");
    }

    // Map from original position (0..N-1) to upper bounds
    SmallVector<Value> originalUBs;
    for (unsigned i = 0; i < numLoops; ++i) {
      scf::ForOp loop = loops[i];
      originalUBs.push_back(loop.getUpperBound());
    }

    // Step 2: Create position symbols for this candidate set
    SmallVector<Value> positions;
    scf::ForOp firstLoop = loops[0];
    Location loc = firstLoop.getLoc();
    OpBuilder builder(firstLoop);
    builder.setInsertionPoint(firstLoop);

    for (unsigned i = 0; i < numLoops; ++i) {
      auto posOp = mlir::ktdf::TilingReserveSizeOp::create(
          builder, loc, builder.getIndexType(), builder.getIndexAttr(0),
          builder.getIndexAttr(1));
      // Tag the symbol so that later passes can recognize it.
      posOp->setAttr(scheduler::kLoopOrderDepthAttr, builder.getIndexAttr(i));
      positions.push_back(posOp.getResult());
    }

    candidateSetPositions[&candidateSet] = positions;

    // Step 3: Create select chains for the loop upperbound, and rewrite upper
    // bounds using select chains.
    auto buildSelectChain = [&](Value posVar, ArrayRef<Value> bounds) -> Value {
      // Step 3a: Create select chains. For loop at position i:
      // select(pos[i]==0, ub[0], select(pos[i]==1, ub[1], ..., ub[N-1]))
      Value result = bounds.back();
      for (int i = static_cast<int>(bounds.size()) - 2; i >= 0; --i) {
        Value bound = bounds[i];
        Value indexConst = arith::ConstantOp::create(
            builder, loc, builder.getIndexType(), builder.getIndexAttr(i));
        Value cond =
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                  posVar, indexConst)
                .getResult();
        result = arith::SelectOp::create(builder, loc, cond, bound, result);
      }
      return result;
    };

    // Step 3b: Apply select chains to upper bounds - each loop's upper bound
    // uses its corresponding position variable. Lower bound and step
    // remain 0 and 1 respectively (unchanged)
    for (unsigned loopIdx = 0; loopIdx < numLoops; ++loopIdx) {
      Value newUB = buildSelectChain(positions[loopIdx], originalUBs);
      scf::ForOp loop = loops[loopIdx];
      loop.setUpperBound(newUB);
    }

    // Step 4: Create selection chains for loop IVs, and replace all uses of IVs
    // in the loop body with the corresponding value from selection chain.
    // Insert arg selection trees in the innermost loop body
    scf::ForOp innermostLoop = loops[numLoops - 1];
    OpBuilder bodyBuilder(innermostLoop.getBody(),
                          innermostLoop.getBody()->begin());

    SmallVector<Value> selectedIVs(numLoops);
    llvm::DenseSet<Operation*> selectionTreeOps;

    // Step 4a: Create IV selection chains. positions[d] holds the index of the
    // original loop that runs at depth d (see the upper bound select chains
    // above). The IV of original loop j is therefore the IV of the depth d for
    // which positions[d] == j.
    for (unsigned origLoopIdx = 0; origLoopIdx < numLoops; ++origLoopIdx) {
      scf::ForOp lastLoop = loops[numLoops - 1];
      Value selectedIV = lastLoop.getInductionVar();
      Value indexConst = arith::ConstantOp::create(
          bodyBuilder, loc, bodyBuilder.getIndexType(),
          bodyBuilder.getIndexAttr(origLoopIdx));
      selectionTreeOps.insert(indexConst.getDefiningOp());
      for (int i = static_cast<int>(numLoops) - 2; i >= 0; --i) {
        scf::ForOp iLoop = loops[i];
        Value iv = iLoop.getInductionVar();

        Value cond =
            arith::CmpIOp::create(bodyBuilder, loc, arith::CmpIPredicate::eq,
                                  positions[i], indexConst)
                .getResult();
        selectionTreeOps.insert(cond.getDefiningOp());

        selectedIV =
            arith::SelectOp::create(bodyBuilder, loc, cond, iv, selectedIV);
        selectionTreeOps.insert(selectedIV.getDefiningOp());
      }
      selectedIVs[origLoopIdx] = selectedIV;
    }

    // Step 4b: Replace uses of original loop IVs with selected IVs (kip uses in
    // selection tree ops and loop definitions)
    llvm::DenseSet<Operation*> loopDefOps;
    for (unsigned i = 0; i < numLoops; ++i) {
      scf::ForOp loop = loops[i];
      loopDefOps.insert(loop.getOperation());
    }

    innermostLoop.getBody()->walk([&](Operation* op) {
      // Skip if this operation is part of selection trees or is a loop
      // definition
      if (selectionTreeOps.count(op) || loopDefOps.count(op)) return;
      if (isa<scf::YieldOp>(op)) return;
      // The operands of buffer_phase are the loops of the pipeline scope in
      // physical nesting order, so they must stay the raw loop IVs.
      if (isa<mlir::ktdf::BufferPhaseOp>(op)) return;

      // Replace uses of original IVs with selected IVs
      for (unsigned loopIdx = 0; loopIdx < numLoops; ++loopIdx) {
        scf::ForOp loop = loops[loopIdx];
        Value origIV = loop.getInductionVar();
        Value selectedIV = selectedIVs[loopIdx];
        for (auto& operand : op->getOpOperands()) {
          if (operand.get() == origIV) {
            operand.set(selectedIV);
          }
        }
      }
    });

    symbolizedSets.push_back(
        {SmallVector<scf::ForOp>(loops.begin(), loops.end()), positions,
         selectedIVs});
  }
  return symbolizedSets;
}

static bool collectTransferDeps(
    mlir::ktdf::DataTransferOp transfer, ArrayRef<scf::ForOp> chain,
    const llvm::DenseMap<Value, std::pair<unsigned, unsigned>>& selectedIVMap,
    llvm::DenseSet<Operation*>& staticLoops,
    llvm::DenseSet<std::pair<unsigned, unsigned>>& setDeps) {
  // Collects what the data in `transfer` depends on, by following the operands
  // of the transfer back to the loop induction variables they are computed
  // from.
  scf::ForOp outermostLoop = chain.front();
  Operation* outermost = outermostLoop.getOperation();
  llvm::DenseSet<Operation*> chainOps;
  for (scf::ForOp loop : chain) chainOps.insert(loop.getOperation());

  SmallVector<Value> worklist(transfer->operand_begin(),
                              transfer->operand_end());
  llvm::DenseSet<Operation*> visited;
  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();

    auto selected = selectedIVMap.find(val);
    if (selected != selectedIVMap.end()) {
      setDeps.insert(selected->second);
      continue;
    }

    if (auto arg = dyn_cast<BlockArgument>(val)) {
      Operation* owner = arg.getOwner()->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(owner);
          forOp && chainOps.count(owner)) {
        if (forOp.getInductionVar() != val) return false;  // iter_arg
        staticLoops.insert(owner);
        continue;
      }
      if (outermost->isProperAncestor(owner)) return false;
      continue;
    }

    Operation* defOp = val.getDefiningOp();
    if (isa<mlir::ktdf::SelectMemrefOp>(defOp)) continue;
    if (!outermost->isProperAncestor(defOp)) continue;
    if (defOp->getNumRegions() != 0) return false;
    if (!visited.insert(defOp).second) continue;
    worklist.append(defOp->operand_begin(), defOp->operand_end());
  }
  return true;
}

static bool conditionalizeTransfer(
    mlir::ktdf::DataTransferOp transfer,
    ArrayRef<SymbolizedSetInfo> symbolizedSets,
    const llvm::DenseMap<Value, std::pair<unsigned, unsigned>>& selectedIVMap,
    const llvm::DenseMap<Operation*, std::pair<unsigned, unsigned>>&
        loopSetInfo) {
  // Step 1: Only transfers between global and ct_local memory -- loads (global
  // -> ct_local) and stores (ct_local -> global) -- are allowed to be
  // conditionalized. Check if the current data transfer is a global <-> local
  // transfer
  auto getMemorySpaceKind =
      [](Value v) -> std::optional<mlir::ktdp::MemorySpaceKind> {
    auto memrefType = dyn_cast<MemRefType>(v.getType());
    if (!memrefType) return std::nullopt;
    auto space = dyn_cast_or_null<mlir::ktdp::MemorySpaceAttr>(
        memrefType.getMemorySpace());
    if (!space) return std::nullopt;
    return space.getKind();
  };
  auto srcKind = getMemorySpaceKind(transfer.getSource());
  auto dstKind = getMemorySpaceKind(transfer.getDestination());
  if (!srcKind || !dstKind) return false;
  bool isLoad = *srcKind == mlir::ktdp::MemorySpaceKind::global &&
                *dstKind == mlir::ktdp::MemorySpaceKind::ct_local;
  bool isStore = *srcKind == mlir::ktdp::MemorySpaceKind::ct_local &&
                 *dstKind == mlir::ktdp::MemorySpaceKind::global;
  if (!isLoad && !isStore) return false;

  auto stage = transfer->getParentOfType<mlir::ktdf::StageOp>();
  if (!stage) return false;
  auto pipeline = dyn_cast<mlir::ktdf::PipelineOp>(stage->getParentOp());
  if (!pipeline) return false;

  // Step 2: Find the loops the transfer is inside of (outermost first).
  SmallVector<scf::ForOp> chain;
  for (Operation* p = transfer->getParentOp(); p; p = p->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(p))
      chain.push_back(forOp);
    else if (isa<LoopLikeOpInterface>(p))
      return false;
  }
  if (chain.empty()) return false;
  std::reverse(chain.begin(), chain.end());
  for (scf::ForOp loop : chain)
    if (!matchPattern(loop.getLowerBound(), m_Zero()) ||
        !matchPattern(loop.getStep(), m_One()))
      return false;
  unsigned n = chain.size();

  // Step 2: Find which loops the transfer depends on. If the transfer contains
  // the IV of a loop in its body, then it depends on that loop and vice-versa
  llvm::DenseSet<Operation*> staticLoops;
  llvm::DenseSet<std::pair<unsigned, unsigned>> setDeps;
  if (!collectTransferDeps(transfer, chain, selectedIVMap, staticLoops,
                           setDeps))
    return false;

  // Step 2a: A transfer that depends on every enclosing loop is never
  // redundant, for any loop ordering, so there is nothing to guard.
  bool dependsOnAllLoops = true;
  for (scf::ForOp loop : chain) {
    if (staticLoops.count(loop.getOperation())) continue;
    auto setInfo = loopSetInfo.find(loop.getOperation());
    if (setInfo == loopSetInfo.end()) {
      dependsOnAllLoops = false;
      break;
    }
    unsigned setIdx = setInfo->second.first;
    for (unsigned j = 0; j < symbolizedSets[setIdx].loops.size(); ++j)
      if (!setDeps.count({setIdx, j})) dependsOnAllLoops = false;
    if (!dependsOnAllLoops) break;
  }
  if (dependsOnAllLoops) return false;

  // Step 2b: Hoisting outside loops present inside a pipeline with modulo > 1
  // requires special handling to ensure all instances are correctly filled.
  // Collect loops enclosing the pipeline whose IVs select the double buffer.
  llvm::DenseSet<Operation*> scopeLoops;
  for (scf::ForOp loop : mlir::ktdf::getPipelineEnclosingScope(pipeline).loops)
    scopeLoops.insert(loop.getOperation());

  // Step 3: For each physical loop, build a condition that is true if the
  // transfer depends on that loop. For loops in a symbolized set, this is true
  // if the transfer depends on the original loop placed at that depth.
  uint64_t numPhases = pipeline.getModulo().value_or(1);
  OpBuilder b(transfer);
  Location loc = transfer.getLoc();
  auto idx = [&](int64_t v) -> Value {
    return arith::ConstantOp::create(b, loc, b.getIndexType(),
                                     b.getIndexAttr(v));
  };
  auto boolConst = [&](bool v) -> Value {
    return arith::ConstantOp::create(b, loc, b.getI1Type(), b.getBoolAttr(v));
  };
  auto andI = [&](Value x, Value y) -> Value {
    return b.createOrFold<arith::AndIOp>(loc, x, y);
  };
  auto orI = [&](Value x, Value y) -> Value {
    return b.createOrFold<arith::OrIOp>(loc, x, y);
  };
  auto cmp = [&](arith::CmpIPredicate pred, Value x, Value y) -> Value {
    return b.createOrFold<arith::CmpIOp>(loc, pred, x, y);
  };
  Value trueV = boolConst(true);
  Value falseV = boolConst(false);

  SmallVector<Value> dep(n, falseV);
  for (unsigned i = 0; i < n; ++i) {
    Operation* loopOp = chain[i].getOperation();
    if (staticLoops.count(loopOp)) {
      dep[i] = trueV;
      continue;
    }
    auto setInfo = loopSetInfo.find(loopOp);
    if (setInfo == loopSetInfo.end()) continue;
    auto [setIdx, depth] = setInfo->second;
    for (auto [depSetIdx, origLoop] : setDeps)
      if (depSetIdx == setIdx)
        dep[i] = orI(dep[i], cmp(arith::CmpIPredicate::eq,
                                 symbolizedSets[setIdx].positions[depth],
                                 idx(origLoop)));
  }

  // Step 4: Find if the transfers can be hoisted outside some of its
  // encompassing loops. Note: transfers will be hoisted only upto the innermost
  // dependent loop. suffixDep[i] is true if the transfer depends on loop i or
  // any inner loop.
  SmallVector<Value> suffixDep(n);
  Value accScope = falseV;
  for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
    Operation* loopOp = chain[i].getOperation();
    if (stage->isProperAncestor(loopOp)) {
      suffixDep[i] = dep[i];
    } else if (scopeLoops.count(loopOp)) {
      accScope = orI(accScope, dep[i]);
      suffixDep[i] = accScope;
    } else {
      suffixDep[i] = trueV;
    }
  }

  // Step 5: Build the condition under which the transfer must run. Loops with
  // suffixDep false are the ones the transfer is redundant across. (1) Loads:
  // only the first iteration of those loops is needed. If the pipeline is
  // double buffered, the first `numPhases` iterations of the loops that select
  // the buffer are needed, so that every buffer is filled. (2) Stores: only the
  // last iteration of those loops is needed, since it holds the final value.
  Value cond = trueV;
  if (isLoad) {
    Value zero = idx(0);
    Value flatIdx = zero;
    Value flatSize = idx(1);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
      Value iv = chain[i].getInductionVar();
      if (numPhases <= 1 || !scopeLoops.count(chain[i].getOperation())) {
        cond = andI(cond,
                    orI(suffixDep[i], cmp(arith::CmpIPredicate::eq, iv, zero)));
        continue;
      }
      Value inFlat = b.createOrFold<arith::XOrIOp>(loc, suffixDep[i], trueV);
      Value newIdx = b.createOrFold<arith::AddIOp>(
          loc, flatIdx, b.createOrFold<arith::MulIOp>(loc, iv, flatSize));
      Value newSize = b.createOrFold<arith::MulIOp>(loc, flatSize,
                                                    chain[i].getUpperBound());
      flatIdx = b.createOrFold<arith::SelectOp>(loc, inFlat, newIdx, flatIdx);
      flatSize =
          b.createOrFold<arith::SelectOp>(loc, inFlat, newSize, flatSize);
    }
    if (numPhases > 1)
      cond = andI(cond, cmp(arith::CmpIPredicate::ult, flatIdx,
                            idx(static_cast<int64_t>(numPhases))));
  } else {
    Value one = idx(1);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
      Value last =
          b.createOrFold<arith::SubIOp>(loc, chain[i].getUpperBound(), one);
      cond =
          andI(cond, orI(suffixDep[i], cmp(arith::CmpIPredicate::eq,
                                           chain[i].getInductionVar(), last)));
    }
  }

  // Step 6: Add the condition around the transfer
  if (matchPattern(cond, m_One())) return false;
  auto ifOp = scf::IfOp::create(b, loc, TypeRange{}, cond,
                                /*withElseRegion=*/false);
  transfer->moveBefore(ifOp.thenBlock()->getTerminator());
  return true;
}

static void conditionalizeDataTransfers(
    ArrayRef<SymbolizedSetInfo> symbolizedSets, Operation* root) {
  // Step 1: Identify all data transfers and collect necessary metadata
  llvm::DenseMap<Value, std::pair<unsigned, unsigned>> selectedIVMap;
  llvm::DenseMap<Operation*, std::pair<unsigned, unsigned>> loopSetInfo;
  for (unsigned s = 0; s < symbolizedSets.size(); ++s) {
    const SymbolizedSetInfo& set = symbolizedSets[s];
    for (unsigned j = 0; j < set.selectedIVs.size(); ++j)
      selectedIVMap[set.selectedIVs[j]] = {s, j};
    for (unsigned d = 0; d < set.loops.size(); ++d) {
      scf::ForOp loop = set.loops[d];
      loopSetInfo[loop.getOperation()] = {s, d};
    }
  }

  SmallVector<mlir::ktdf::DataTransferOp> transfers;
  root->walk([&](mlir::ktdf::DataTransferOp transfer) {
    transfers.push_back(transfer);
  });

  // Step 2: Check if each data transfer can be conditionalized. If yes, put in
  // the correct condition to avoid redundant transfers
  unsigned numConditionalized = 0;
  for (mlir::ktdf::DataTransferOp transfer : transfers)
    if (conditionalizeTransfer(transfer, symbolizedSets, selectedIVMap,
                               loopSetInfo))
      ++numConditionalized;

#ifdef DEBUG_LOOP_TILING
  llvm::outs() << "[SymbolizeLoopOrderAndDataTransfer] Made "
               << numConditionalized << " of " << transfers.size()
               << " data transfer(s) conditional.\n";
  llvm::outs().flush();
#endif
  (void)numConditionalized;
}

struct SymbolizeLoopOrderAndDataTransferPass
    : public scheduler::impl::SymbolizeLoopOrderAndDataTransferPassBase<
          SymbolizeLoopOrderAndDataTransferPass> {
  using SymbolizeLoopOrderAndDataTransferPassBase<
      SymbolizeLoopOrderAndDataTransferPass>::
      SymbolizeLoopOrderAndDataTransferPassBase;

  void runOnOperation() override {
    Operation* module = getOperation();

    // Step 1: Identify candidate sets of loops for reordering. All loops that
    // fall into a set can be ordered arbitrarily. If ops in between loops
    // create false barriers, they are hoisted/sunk to allow reordering
    SmallVector<CandidateSetInfo> candidateSets;
    findCandidateSets(module, candidateSets);

#ifdef DEBUG_LOOP_TILING
    llvm::outs() << "[SymbolizeLoopOrderAndDataTransfer] IR after "
                    "findCandidateSets():\n";
    module->print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs().flush();
#endif

#ifdef DEBUG_LOOP_TILING
    if (candidateSets.empty()) {
      llvm::outs() << "[SymbolizeLoopOrderAndDataTransfer] No "
                      "interchange-candidate loop sets found.\n";
      llvm::outs().flush();
      return;
    }

    llvm::outs() << "[SymbolizeLoopOrderAndDataTransfer] Found "
                 << candidateSets.size()
                 << " interchange-candidate loop set(s):";
    for (auto [idx, info] : llvm::enumerate(candidateSets))
      printCandidateSet(static_cast<unsigned>(idx), info.loops, info.pairOps,
                        llvm::outs());
    llvm::outs() << "\n";
    llvm::outs().flush();
#endif

    // Step 2: Symbolize loop ordering for each candidate set
    SmallVector<SymbolizedSetInfo> symbolizedSets =
        symbolizeLoopOrdering(candidateSets);

    // Step 3: Make data transfers conditional on the loop ordering, to avoid
    // redundant transfers
    conditionalizeDataTransfers(symbolizedSets, module);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createSymbolizeLoopOrderAndDataTransferPass() {
  return std::make_unique<SymbolizeLoopOrderAndDataTransferPass>();
}
