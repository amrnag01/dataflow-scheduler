//===-- TileSizeSelection.cpp -----------------------------------*- c++ -*-===//
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
// Pass: -tile-size-selection
//
// Resolve ktdf.tiling.reserve_size placeholders to concrete index typed SSA
// values.
//
//===----------------------------------------------------------------------===//

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/TileSizeApply.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/TileSizeInfo.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "dataflow-scheduler/Utils/TileSizeAgentInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "tile-size-selection"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;
using namespace scheduler;

namespace mlir::ktdf {
#define GEN_PASS_DEF_TILESIZESELECTIONPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

namespace {


void logUnresolved(ktdf::TilingReserveSizeOp reserve_size_op,
                   llvm::StringRef reason) {
  LDBG(1) << "unresolved tiling.reserve_size at " << reserve_size_op.getLoc()
          << ": " << reason;
}

void collectAssociatedLoops(
    Value value, TileSizeInfo& ts_info, llvm::DenseSet<Value>& visited_values,
    SmallVectorImpl<ktdf::TilingReserveSizeOp>& unresolved_ops) {
  if (!visited_values.insert(value).second) return;

  for (Operation* user : value.getUsers()) {
    // Pattern: %bound = arith.ceildivui %total_size, %tile_size
    // Then: scf.for %i = %c0 to %bound step %c1
    if (auto ceildiv_op = dyn_cast<arith::CeilDivUIOp>(user)) {
      // Check if tile_size is the divisor (RHS)
      if (ceildiv_op.getRhs() != value) {
        logUnresolved(ts_info.reserve_size_op,
                      "tile size used as dividend (LHS) in ceildivui, expected "
                      "divisor (RHS)");
        unresolved_ops.push_back(ts_info.reserve_size_op);
        continue;
      }

      // Get the total size (LHS of ceildiv)
      auto total_size_opt =
          scheduler::getConstantIndexValue(ceildiv_op.getLhs());
      if (!total_size_opt.has_value()) {
        logUnresolved(ts_info.reserve_size_op,
                      "non-constant total size in arith.ceildivui");
        unresolved_ops.push_back(ts_info.reserve_size_op);
        continue;
      }

      // Now trace the ceildiv result to find loops that use it as upper bound
      Value ceildiv_result = ceildiv_op.getResult();
      for (Operation* bound_user : ceildiv_result.getUsers()) {
        if (auto for_op = dyn_cast<scf::ForOp>(bound_user)) {
          if (for_op.getUpperBound() == ceildiv_result) {
            ts_info.associated_loops.push_back(
                AssociatedLoopInfo{for_op, *total_size_opt});
          }
        }
      }
      continue;
    }

    // TODO: in future consider memref.alloc sizes to constrain the tile size
    // selection
  }
}

struct TileSizeSelectionPass
    : public ktdf::impl::TileSizeSelectionPassBase<TileSizeSelectionPass> {
  const SchedulerExtContext* scheduler_ctx = nullptr;

  void runOnOperation() override {
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp module = getOperation();

    if (!scheduler_ctx) {
      llvm::report_fatal_error(
          "TileSizeSelectionPass requires SchedulerExtContext");
    }

    SmallVector<ktdf::TilingReserveSizeOp> reserve_size_ops;
    module.walk([&](ktdf::TilingReserveSizeOp reserve_size_op) {
      reserve_size_ops.push_back(reserve_size_op);
    });

    SmallVector<TileSizeInfo> analyses;
    analyses.reserve(reserve_size_ops.size());

    SmallVector<ktdf::TilingReserveSizeOp> unresolved_ops;
    for (ktdf::TilingReserveSizeOp reserve_size_op : reserve_size_ops) {
      TileSizeInfo ts_info;
      ts_info.reserve_size_op = reserve_size_op;
      llvm::DenseSet<Value> visited_values;
      collectAssociatedLoops(reserve_size_op.getResult(), ts_info,
                             visited_values, unresolved_ops);
      analyses.push_back(std::move(ts_info));
    }

    // Use agentic loop to select all tile sizes at once
    std::vector<int64_t> chosen_tile_sizes =
        const_cast<SchedulerExtContext*>(scheduler_ctx)->selectAllTileSizes(module, analyses);

    if (chosen_tile_sizes.size() != analyses.size()) {
      llvm::report_fatal_error(
          "Agentic selector returned wrong number of tile sizes");
    }

    // Validate each tile size
    for (size_t i = 0; i < analyses.size(); ++i) {
      TileSizeInfo& ts_info = analyses[i];
      int64_t tile_size = chosen_tile_sizes[i];

      int64_t min_value = ts_info.reserve_size_op.getMinValue().getSExtValue();
      int64_t divisibility =
          ts_info.reserve_size_op.getDivisibility().getSExtValue();

      if (!validateTileSizeResult(tile_size, ts_info, min_value, divisibility)) {
        llvm::report_fatal_error(
            llvm::Twine("Agent produced invalid tile size: ") +
            llvm::Twine(tile_size));
      }
    }

    // Apply tile sizes
    OpBuilder builder(module.getContext());
    for (size_t i = 0; i < analyses.size(); ++i) {
      applyTileSize(builder, analyses[i], chosen_tile_sizes[i]);
    }
  }
};

}  // namespace

auto mlir::ktdf::createTileSizeSelectionPass(
    const scheduler::SchedulerExtContext& scheduler_ctx) -> std::unique_ptr<Pass> {
  auto pass = std::make_unique<TileSizeSelectionPass>();
  pass->scheduler_ctx = &scheduler_ctx;
  return pass;
}
