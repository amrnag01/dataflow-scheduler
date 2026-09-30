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
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
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

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Tile Size Selection pass"),
    llvm::cl::init(false));

static llvm::cl::list<int64_t> ClTileSizes(
    "tile-sizes",
    llvm::cl::desc("Tile sizes for each reserved dimension (global override)"),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);

using namespace mlir;

namespace mlir::ktdf {
#define GEN_PASS_DEF_TILESIZESELECTIONPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

namespace {

// TODO: should use much large size when proper L1 usage analysis is available.
constexpr int64_t kMaxCandidateTileSize = 2;

struct AssociatedLoopInfo {
  scf::ForOp loop;

  // The total size being tiled (numerator in ceildiv operation).
  // For pattern: %bound = arith.ceildivui %total_size, %tile_size
  // This represents %total_size which must be evenly divisible by the tile
  // size.
  int64_t total_size;
};

struct TileSizeInfo {
  ktdf::TilingReserveSizeOp reserve_size_op;
  SmallVector<AssociatedLoopInfo> associated_loops;
};

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
      auto total_size_opt = getConstantIntValue(ceildiv_op.getLhs());
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

enum class TileSizeValidity {
  // Candidate tile size is valid for all loops
  kValidForAllLoops,
  // Invalid for this candidate tile size.
  kInvalidForThisTileSize,
};

TileSizeValidity evaluateCandidateTileSize(const TileSizeInfo& ts_info,
                                           int64_t candidate) {
  for (const AssociatedLoopInfo& loop_info : ts_info.associated_loops) {
    // Check if the total size is evenly divisible by the candidate tile size
    if (loop_info.total_size % candidate != 0) {
      return TileSizeValidity::kInvalidForThisTileSize;
    }
  }

  return TileSizeValidity::kValidForAllLoops;
}

std::optional<int64_t> chooseTileSize(
    TileSizeInfo& ts_info,
    SmallVectorImpl<ktdf::TilingReserveSizeOp>& unresolved_ops) {
  const int64_t min_value =
      ts_info.reserve_size_op.getMinValue().getSExtValue();
  const int64_t divisibility =
      ts_info.reserve_size_op.getDivisibility().getSExtValue();

  if (ts_info.associated_loops.empty()) {
    logUnresolved(ts_info.reserve_size_op,
                  "no associated loops found via ceildivui pattern");
    unresolved_ops.push_back(ts_info.reserve_size_op);
    return std::nullopt;
  }

  const int64_t start_candidate =
      std::max<int64_t>(kMaxCandidateTileSize, min_value);
  for (int64_t candidate = start_candidate; candidate >= min_value;
       --candidate) {
    LLVM_DEBUG({
      if (candidate <= 1) {
        llvm::dbgs() << "[" PASS_NAME
                     << "] no suitable tile size greater than one for "
                        "tiling.reserve_size at "
                     << ts_info.reserve_size_op.getLoc() << ".\n";
      }
    });
    if (divisibility > 1 && candidate % divisibility != 0) continue;
    TileSizeValidity eval = evaluateCandidateTileSize(ts_info, candidate);
    if (eval == TileSizeValidity::kValidForAllLoops) return candidate;
  }
  // At the very least a tile size of 1 should have been chosen.
  llvm_unreachable("unresolved tile size");
  return std::nullopt;
}

struct TileSizeSelectionPass
    : public ktdf::impl::TileSizeSelectionPassBase<TileSizeSelectionPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";
    ModuleOp module = getOperation();

    SmallVector<ktdf::TilingReserveSizeOp> reserve_size_ops;
    module.walk([&](ktdf::TilingReserveSizeOp reserve_size_op) {
      reserve_size_ops.push_back(reserve_size_op);
    });

    if (reserve_size_ops.empty()) return;

    ArrayRef<int64_t> effective_tile_sizes(ClTileSizes);

    if (!effective_tile_sizes.empty() &&
        effective_tile_sizes.size() != reserve_size_ops.size()) {
      module.emitError() << "[" PASS_NAME
                         << "] Number of tile sizes specified ("
                         << effective_tile_sizes.size()
                         << ") does not match the number of reserve_size ops ("
                         << reserve_size_ops.size() << ")";
      return signalPassFailure();
    }

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

    OpBuilder builder(module.getContext());
    for (auto [idx, ts_info] : llvm::enumerate(analyses)) {
      std::optional<int64_t> chosen_tile_size;
      if (!effective_tile_sizes.empty()) {
        chosen_tile_size = effective_tile_sizes[idx];
      } else {
        chosen_tile_size = chooseTileSize(ts_info, unresolved_ops);
      }
      if (!chosen_tile_size.has_value()) {
        continue;
      }

      builder.setInsertionPoint(ts_info.reserve_size_op);
      auto constant_op = arith::ConstantIndexOp::create(
          builder, ts_info.reserve_size_op.getLoc(), *chosen_tile_size);
      ts_info.reserve_size_op.getResult().replaceAllUsesWith(
          constant_op.getResult());
      ts_info.reserve_size_op.erase();
    }

    // TODO: Replace this placeholder heuristic with a real cost model that
    // accounts for L1 memory usage and performance. In particular, a future
    // liveness analysis should determine which L1 buffers are live at each
    // point so we can compute peak live scratchpad usage and choose tile sizes
    // that fully utilize L1 capacity without exceeding the hardware limit.
    (void)unresolved_ops;
  }
};

}  // namespace

auto mlir::ktdf::createTileSizeSelectionPass() -> std::unique_ptr<Pass> {
  return std::make_unique<TileSizeSelectionPass>();
}
