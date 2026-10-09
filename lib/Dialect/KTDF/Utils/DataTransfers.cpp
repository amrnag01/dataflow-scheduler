//===-- DataTransfers.cpp ---------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Transforms/Utils/DataTransfers.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"

using namespace scheduler;

namespace {

using MemRef = mlir::TypedValue<mlir::MemRefType>;

[[nodiscard]] auto createMap(mlir::OpBuilder& builder,
                             mlir::OffsetSizeAndStrideOpInterface iface)
    -> std::tuple<mlir::AffineMap, mlir::ValueRange,
                  mlir::SmallVector<mlir::OpFoldResult>> {
  llvm::SmallVector<mlir::AffineExpr> results;

  auto offsets = iface.getMixedOffsets();
  auto strides = iface.getMixedStrides();
  unsigned dim_idx = 0;
  for (auto idx : llvm::iota_range<unsigned>(0, offsets.size(), false)) {
    if (const auto value = mlir::getConstantIntValue(offsets[idx]); value) {
      results.push_back(builder.getAffineConstantExpr(*value));
    } else {
      results.push_back(builder.getAffineDimExpr(dim_idx++));
    }

    if (const auto value = mlir::getConstantIntValue(strides[idx]); value) {
      results.back() = results.back() * builder.getAffineConstantExpr(*value);
    } else {
      llvm::report_fatal_error("dynamic strides are not supported");
    }
  }

  return {mlir::AffineMap::get(iface.getOffsets().size(), 0, results,
                               builder.getContext()),
          iface.getOffsets(), iface.getMixedSizes()};
}

struct LowerLoadToFifo : mlir::OpRewritePattern<mlir::ktdf::WriteToFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdf::WriteToFifoOp write,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto bcast = write.getData().getDefiningOp<mlir::tensor::ExtractSliceOp>();
    auto load =
        bcast ? bcast.getSource().getDefiningOp<mlir::ktdp_lowering::LoadOp>()
              : write.getData().getDefiningOp<mlir::ktdp_lowering::LoadOp>();
    if (!load) {
      return rewriter.notifyMatchFailure(write, "does not read from memory");
    }
    auto memref = mlir::dyn_cast<MemRef>(load.getSource());
    if (!memref) {
      return rewriter.notifyMatchFailure(load, "not bufferized");
    }

    const auto [source_map, source_offsets, source_sizes] =
        createMap(rewriter, load);
    llvm::SmallVector<mlir::OpFoldResult> dest_sizes(source_sizes);
    if (bcast) {
      const auto [dest_map, dest_offsets, sizes] = createMap(rewriter, bcast);
      if (!dest_map.isConstant() ||
          llvm::count(dest_map.getConstantResults(), 0) !=
              dest_map.getNumResults()) {
        return rewriter.notifyMatchFailure(bcast, "broadcast not legalizble");
      }
      dest_sizes = sizes;
    }
    auto transfer = mlir::ktdf::DataTransferOp::create(
        rewriter, rewriter.getFusedLoc({load.getLoc(), write.getLoc()}), memref,
        source_map, source_offsets, source_sizes, write.getFifoSlot(), {}, {},
        dest_sizes);
    transfer->setDiscardableAttrs(load->getRawDictionaryAttrs());
    rewriter.eraseOp(write);
    return llvm::success();
  }
};

struct LowerStoreFromFifo
    : mlir::OpRewritePattern<mlir::ktdp_lowering::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdp_lowering::StoreOp store,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto read = store.getSource().getDefiningOp<mlir::ktdf::ReadFromFifoOp>();
    if (!read) {
      return rewriter.notifyMatchFailure(store, "does not read from FIFO");
    }
    if (!read->hasOneUse()) {
      return read.emitError("has multiple uses");
    }
    auto memref = mlir::dyn_cast<MemRef>(store.getDest());
    if (!memref) {
      return rewriter.notifyMatchFailure(store, "not bufferized");
    }

    auto [map, ivs, sizes] = createMap(rewriter, store);
    auto transfer = mlir::ktdf::DataTransferOp::create(
        rewriter, rewriter.getFusedLoc({read.getLoc(), store.getLoc()}),
        read.getFifoSlot(), {}, {}, sizes, memref, map, ivs, sizes);
    transfer->setDiscardableAttrs(store->getRawDictionaryAttrs());
    rewriter.eraseOp(store);
    rewriter.eraseOp(read);
    return llvm::success();
  }
};

struct LowerStoreFromLoad
    : mlir::OpRewritePattern<mlir::ktdp_lowering::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdp_lowering::StoreOp store,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto source = llvm::dyn_cast<mlir::OpResult>(store.getSource());
    if (!source) {
      return rewriter.notifyMatchFailure(store, "does not load from memory");
    }

    // Ignore any sequence of `tensor.(collapse|expand)_shape`, since these are
    // order-preserving and thus folded into the `ktdf.data_transfer`.
    auto load = llvm::dyn_cast<mlir::ktdp_lowering::LoadOp>(source.getOwner());
    while (!load) {
      if (auto collapse =
              llvm::dyn_cast<mlir::tensor::CollapseShapeOp>(source.getOwner());
          collapse) {
        source = llvm::dyn_cast<mlir::OpResult>(collapse.getSrc());
        if (!source) {
          return rewriter.notifyMatchFailure(collapse,
                                             "does not load from memory");
        }
      } else if (auto expand = llvm::dyn_cast<mlir::tensor::ExpandShapeOp>(
                     source.getOwner());
                 expand) {
        source = llvm::dyn_cast<mlir::OpResult>(expand.getSrc());
        if (!source) {
          return rewriter.notifyMatchFailure(expand,
                                             "does not load from memory");
        }
      } else {
        return rewriter.notifyMatchFailure(source.getOwner(),
                                           "does not load from memory");
      }

      load = llvm::dyn_cast<mlir::ktdp_lowering::LoadOp>(source.getOwner());
    }

    auto [load_map, load_ivs, load_sizes] = createMap(rewriter, load);
    auto [store_map, store_ivs, store_sizes] = createMap(rewriter, store);
    auto transfer = mlir::ktdf::DataTransferOp::create(
        rewriter, rewriter.getFusedLoc({load.getLoc(), store.getLoc()}),
        load.getSource(), load_map, load_ivs, load_sizes, store.getDest(),
        store_map, store_ivs, store_sizes);
    transfer->setDiscardableAttrs(store->getRawDictionaryAttrs());
    rewriter.eraseOp(store);
    return llvm::success();
  }
};

}  // namespace

void scheduler::populateConvertToDataTransferPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<LowerLoadToFifo, LowerStoreFromFifo, LowerStoreFromLoad>(
      patterns.getContext());
}
