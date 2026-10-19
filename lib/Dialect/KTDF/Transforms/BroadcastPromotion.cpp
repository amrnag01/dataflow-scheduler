//===-- BroadcastPromotion.cpp ----------------------------------*- c++ -*-===//
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
// BroadcastPromotion: hoist invariant ktdf.data_transfer ops as sibling
// pipelines. See
// docs/superpowers/specs/2026-05-12-broadcast-promotion-design.md
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/Reuse.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/RegionClonePrune.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "broadcast-promotion"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Broadcast Promotion pass"),
    llvm::cl::init(false));

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_BROADCASTPROMOTIONPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

namespace {

using mlir::ktdf::reuse::findUnderlyingAlloc;

/// Clone the defining memref.alloc of `orig_dest` immediately before
/// `insert_before_op`. Returns the cloned alloc's SSA result value.
/// Handles the case where `orig_dest` is a result of ktdf.private by
/// finding the underlying alloc inside the private body.
/// Returns failure and emits an error on `insert_before_op` if no alloc
/// can be found.
FailureOr<Value> cloneDestAllocAbove(Value orig_dest,
                                     Operation* insert_before_op) {
  memref::AllocOp alloc = findUnderlyingAlloc(orig_dest);
  if (!alloc)
    return insert_before_op->emitError(
        "broadcast promotion: destination does not trace back to a "
        "memref.alloc");
  OpBuilder builder(insert_before_op);
  Operation* new_alloc = builder.clone(*alloc.getOperation());
  return new_alloc->getResult(0);
}

/// Perform one broadcast hoist for `c`.
LogicalResult performHoist(const ktdf::reuse::Candidate& c) {
  ktdf::DataTransferOp transfer = c.transfer;
  ktdf::StageOp donor_stage = c.donor_stage;
  scf::ForOp target_loop = c.target_loop;
  Value orig_dest = transfer.getDestination();

  LDBG(1) << " hoisting " << *transfer.getOperation() << "\n";

  // 1. Fresh destination alloc immediately before target_loop.
  FailureOr<Value> new_dest_or =
      cloneDestAllocAbove(orig_dest, target_loop.getOperation());
  if (failed(new_dest_or)) return failure();
  Value new_dest = *new_dest_or;

  // 2. Sibling pipeline + stage immediately after the new alloc, still
  //    before target_loop.
  OpBuilder builder(target_loop);
  auto new_pipeline = ktdf::PipelineOp::create(builder, target_loop.getLoc());
  builder.setInsertionPointToStart(new_pipeline.getBody());
  auto new_stage = ktdf::StageOp::create(builder, target_loop.getLoc(),
                                         /*depends_in=*/ValueRange{},
                                         /*depends_out=*/ValueRange{});
  if (auto applicable_units = donor_stage.getApplicableUnitsAttr()) {
    new_stage.setApplicableUnitsAttr(applicable_units);
  }

  // 3. Seed IRMapping: original dest -> new_dest.
  IRMapping value_map;
  value_map.map(orig_dest, new_dest);

  // 4. Clone donor stage body into new stage body; prune to the transfer.
  Block* new_stage_body = new_stage.getBody();
  scheduler::cloneRegionAndPruneToAnchor(donor_stage.getBodyRegion(),
                                         new_stage_body, new_stage_body->end(),
                                         transfer.getOperation(), value_map);

  // 5. Rewire uses of orig_dest outside donor_stage to new_dest.
  orig_dest.replaceUsesWithIf(new_dest, [&](OpOperand& use) {
    Operation* user = use.getOwner();
    return !donor_stage.getOperation()->isAncestor(user);
  });

  // 6. Erase original transfer.
  transfer.erase();

  // NOTE: if the donor stage's body becomes empty after erasing the
  // transfer, the in-memory IR is well-formed but the printer/parser
  // round-trip fails (an empty ktdf.stage body without explicit ^bb0:
  // re-parses as a 0-block region, which fails the SizedRegion<1>
  // verifier). This is a dialect-level printer/parser asymmetry, not
  // a pass bug. Round-trip tests that pipe pass output through
  // scheduler again will hit it. Tracking as a known limitation.
  return success();
}

/// Walk pipelines in post-order; sets `did_hoist` to true iff a hoist was
/// performed. Returns failure if the hoist itself failed.
LogicalResult tryHoistOneBroadcast(ModuleOp module, bool& did_hoist) {
  did_hoist = false;
  LogicalResult result = success();
  module.walk<WalkOrder::PostOrder>(
      [&](ktdf::PipelineOp pipeline) -> WalkResult {
        if (auto c = ktdf::reuse::findFirstCandidate(pipeline)) {
          if (failed(performHoist(*c))) {
            result = failure();
            return WalkResult::interrupt();
          }
          did_hoist = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  return result;
}

struct BroadcastPromotionPass
    : public ktdf::impl::BroadcastPromotionPassBase<BroadcastPromotionPass> {
  using BroadcastPromotionPassBase<
      BroadcastPromotionPass>::BroadcastPromotionPassBase;

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp module = getOperation();
    bool changed = true;
    while (changed) {
      if (failed(tryHoistOneBroadcast(module, changed)))
        return signalPassFailure();
    }
  }
};

}  // namespace

auto mlir::ktdf::createBroadcastPromotionPass() -> std::unique_ptr<Pass> {
  return std::make_unique<BroadcastPromotionPass>();
}
