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
// Pass: -optimize-loop-order-and-tiling
//
// Chooses values for the symbols (ktdf.tiling.reserve_size ops) left in the IR
// by -symbolize-loop-order-and-data-transfer, i.e. the loop order of each
// candidate set and the tile sizes, and applies them to the IR.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/LoopOrderSymbols.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "optimize-loop-order-and-tiling"

using namespace mlir;

namespace scheduler {
#define GEN_PASS_DEF_OPTIMIZELOOPORDERANDTILINGPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Applies values to symbols in the IR. Each key of `symbolValues` must be the
/// result of a ktdf.tiling.reserve_size op (a tile size or a loop position
/// created by -symbolize-loop-order-and-data-transfer). It is replaced with an
/// index constant holding the given value, and the reserve_size op is erased.
/// Fails, leaving the IR unchanged, if a key is not such a result or its value
/// violates the min_value or divisibility of the symbol.
static LogicalResult resolveSymbols(
    const llvm::DenseMap<Value, int64_t>& symbolValues) {
  // Step 1: Check every symbol before changing anything.
  SmallVector<std::pair<mlir::ktdf::TilingReserveSizeOp, int64_t>> resolved;
  for (auto [symbol, value] : symbolValues) {
    auto reserveOp = symbol.getDefiningOp<mlir::ktdf::TilingReserveSizeOp>();
    if (!reserveOp) {
      mlir::emitError(symbol.getLoc())
          << "resolveSymbols: value is not the result of a "
             "ktdf.tiling.reserve_size";
      return failure();
    }
    int64_t minValue = reserveOp.getMinValue().getSExtValue();
    int64_t divisibility = reserveOp.getDivisibility().getSExtValue();
    if (value < minValue) {
      reserveOp.emitError() << "resolveSymbols: value " << value
                            << " is below the symbol's min_value " << minValue;
      return failure();
    }
    if (divisibility > 1 && value % divisibility != 0) {
      reserveOp.emitError()
          << "resolveSymbols: value " << value
          << " is not divisible by the symbol's divisibility " << divisibility;
      return failure();
    }
    resolved.push_back({reserveOp, value});
  }

  // Step 2: Replace each symbol with a constant.
  for (auto [reserveOp, value] : resolved) {
    OpBuilder builder(reserveOp);
    Value constant =
        arith::ConstantIndexOp::create(builder, reserveOp.getLoc(), value);
    reserveOp.getResult().replaceAllUsesWith(constant);
    reserveOp.erase();
  }
  return success();
}

// ---------------------------------------------------------------------------

/// Test input for resolveSymbols(). Every tile size is 2, and every candidate
/// set keeps its original loop order, i.e. positions[d] = d.
static llvm::DenseMap<Value, int64_t> makeTestSymbolValues(Operation* root) {
  llvm::DenseMap<Value, int64_t> symbolValues;
  root->walk([&](mlir::ktdf::TilingReserveSizeOp reserveOp) {
    // Loop position symbols are tagged with their depth in the candidate set.
    if (auto depth = reserveOp->getAttrOfType<IntegerAttr>(
            scheduler::kLoopOrderDepthAttr))
      symbolValues[reserveOp.getResult()] = depth.getInt();
    else  // Everything else is a tile size.
      symbolValues[reserveOp.getResult()] = 2;
  });
  return symbolValues;
}

// ---------------------------------------------------------------------------

struct OptimizeLoopOrderAndTilingPass
    : public scheduler::impl::OptimizeLoopOrderAndTilingPassBase<
          OptimizeLoopOrderAndTilingPass> {
  using OptimizeLoopOrderAndTilingPassBase<
      OptimizeLoopOrderAndTilingPass>::OptimizeLoopOrderAndTilingPassBase;

  void runOnOperation() override {
    Operation* module = getOperation();

    // Step 1: Apply values to the symbols. The values are not chosen yet, so
    // a fixed test input is used for now.
    llvm::DenseMap<Value, int64_t> symbolValues = makeTestSymbolValues(module);
    if (failed(resolveSymbols(symbolValues))) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createOptimizeLoopOrderAndTilingPass() {
  return std::make_unique<OptimizeLoopOrderAndTilingPass>();
}
