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

#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"

using namespace scheduler;

namespace {

using MemRef = mlir::TypedValue<mlir::MemRefType>;

/// Builds the access map of @p iface : constant offsets become constant
/// results, every other offset becomes a dim. Returns the map, the values bound
/// to its dims, and the sizes.
[[nodiscard]] auto createMap(mlir::OpBuilder& builder,
                             mlir::OffsetSizeAndStrideOpInterface iface)
    -> std::tuple<mlir::AffineMap, mlir::SmallVector<mlir::Value>,
                  mlir::SmallVector<mlir::OpFoldResult>> {
  llvm::SmallVector<mlir::AffineExpr> results;
  llvm::SmallVector<mlir::Value> dims;

  auto offsets = iface.getMixedOffsets();
  auto strides = iface.getMixedStrides();
  unsigned dim_idx = 0;
  for (auto idx : llvm::iota_range<unsigned>(0, offsets.size(), false)) {
    if (const auto value = mlir::getConstantIntValue(offsets[idx]); value) {
      results.push_back(builder.getAffineConstantExpr(*value));
    } else {
      results.push_back(builder.getAffineDimExpr(dim_idx++));
      dims.push_back(llvm::cast<mlir::Value>(offsets[idx]));
    }

    if (const auto value = mlir::getConstantIntValue(strides[idx]); value) {
      results.back() = results.back() * builder.getAffineConstantExpr(*value);
    } else {
      llvm::report_fatal_error("dynamic strides are not supported");
    }
  }

  return {mlir::AffineMap::get(dims.size(), 0, results, builder.getContext()),
          dims, iface.getMixedSizes()};
}

/// Finds the `ktdf.read_from_fifo` that produces @p value , looking through
/// `tensor.(collapse|expand)_shape` ops, which are order-preserving and thus
/// folded into the transfer. Collects those ops into @p reshapes , from
/// @p value towards the read.
[[nodiscard]] auto findFifoRead(
    mlir::Value value, llvm::SmallVectorImpl<mlir::Operation*>& reshapes)
    -> mlir::ktdf::ReadFromFifoOp {
  while (auto* const op = value.getDefiningOp()) {
    if (auto read = llvm::dyn_cast<mlir::ktdf::ReadFromFifoOp>(op); read) {
      return read;
    }
    if (!llvm::isa<mlir::tensor::CollapseShapeOp, mlir::tensor::ExpandShapeOp>(
            op)) {
      return nullptr;
    }
    reshapes.push_back(op);
    value = op->getOperand(0);
  }

  return nullptr;
}

/// Erases @p store , then the @p reshapes and the @p read that fed it.
void eraseFifoRead(mlir::PatternRewriter& rewriter, mlir::Operation* store,
                   llvm::ArrayRef<mlir::Operation*> reshapes,
                   mlir::ktdf::ReadFromFifoOp read) {
  rewriter.eraseOp(store);
  for (auto* const reshape : reshapes) {
    rewriter.eraseOp(reshape);
  }
  rewriter.eraseOp(read);
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
    llvm::SmallVector<mlir::Operation*> reshapes;
    auto read = findFifoRead(store.getSource(), reshapes);
    if (!read) {
      return rewriter.notifyMatchFailure(store, "does not read from FIFO");
    }
    if (!read->hasOneUse()) {
      return read.emitError("has multiple uses");
    }
    if (!llvm::all_of(reshapes, [](auto* op) { return op->hasOneUse(); })) {
      return rewriter.notifyMatchFailure(store, "reshape has multiple uses");
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
    eraseFifoRead(rewriter, store, reshapes, read);
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

/// Converts `ktdf.write_to_fifo(ktdp_lowering.ind_load)` to a gather
/// `ktdf.ind_data_transfer` from the base memref into the FIFO slot.
struct LowerIndLoadToFifo : mlir::OpRewritePattern<mlir::ktdf::WriteToFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdf::WriteToFifoOp write,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto load = write.getData().getDefiningOp<mlir::ktdp_lowering::IndLoadOp>();
    if (!load) {
      return rewriter.notifyMatchFailure(
          write, "does not read indirectly from memory");
    }
    const auto type = load.getResult().getType();
    if (!type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(load, "expected static shape");
    }

    const auto [source_map, source_offsets, source_sizes] =
        createMap(rewriter, load);
    auto transfer = mlir::ktdf::IndDataTransferOp::create(
        rewriter, rewriter.getFusedLoc({load.getLoc(), write.getLoc()}),
        load.getIndAddrBuf(), load.getIndAddrBufIndex(), load.getBase(),
        source_map, source_offsets, source_sizes, mlir::Value{}, mlir::Value{},
        write.getFifoSlot(), mlir::AffineMap{}, mlir::ValueRange{},
        mlir::getAsIndexOpFoldResult(rewriter.getContext(), type.getShape()));
    transfer->setDiscardableAttrs(load->getRawDictionaryAttrs());
    rewriter.eraseOp(write);
    return llvm::success();
  }
};

/// Converts `ktdp_lowering.ind_store(ktdf.read_from_fifo)` to a scatter
/// `ktdf.ind_data_transfer` from the FIFO slot into the base memref.
struct LowerIndStoreFromFifo
    : mlir::OpRewritePattern<mlir::ktdp_lowering::IndStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdp_lowering::IndStoreOp store,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    llvm::SmallVector<mlir::Operation*> reshapes;
    auto read = findFifoRead(store.getSource(), reshapes);
    if (!read) {
      return rewriter.notifyMatchFailure(store, "does not read from FIFO");
    }
    if (!read->hasOneUse()) {
      return read.emitError("has multiple uses");
    }
    if (!llvm::all_of(reshapes, [](auto* op) { return op->hasOneUse(); })) {
      return rewriter.notifyMatchFailure(store, "reshape has multiple uses");
    }
    const auto type = store.getSource().getType();
    if (!type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(store, "expected static shape");
    }

    const auto [dest_map, dest_offsets, dest_sizes] =
        createMap(rewriter, store);
    auto transfer = mlir::ktdf::IndDataTransferOp::create(
        rewriter, rewriter.getFusedLoc({read.getLoc(), store.getLoc()}),
        mlir::Value{}, mlir::Value{}, read.getFifoSlot(), mlir::AffineMap{},
        mlir::ValueRange{},
        mlir::getAsIndexOpFoldResult(rewriter.getContext(), type.getShape()),
        store.getIndAddrBuf(), store.getIndAddrBufIndex(), store.getBase(),
        dest_map, dest_offsets, dest_sizes);
    transfer->setDiscardableAttrs(store->getRawDictionaryAttrs());
    eraseFifoRead(rewriter, store, reshapes, read);
    return llvm::success();
  }
};

/// Folds a `tensor.(collapse|expand)_shape` of a `ktdf.read_from_fifo` into the
/// read: a FIFO slot is flat, so the read may produce the reshaped type
/// directly. Only applies when the reshape is the read's sole user, since a
/// FIFO slot cannot be read twice.
struct FoldReshapeOfFifoRead
    : mlir::OpRewritePattern<mlir::ktdf::ReadFromFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdf::ReadFromFifoOp read,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    if (!read->hasOneUse()) {
      return rewriter.notifyMatchFailure(read, "expected a single use");
    }
    auto* const reshape = *read->getUsers().begin();
    if (!llvm::isa<mlir::tensor::CollapseShapeOp, mlir::tensor::ExpandShapeOp>(
            reshape)) {
      return rewriter.notifyMatchFailure(read, "not read by a reshape");
    }

    // Retyping in place keeps the read's attributes (e.g. `splat`).
    rewriter.modifyOpInPlace(read, [&]() {
      read.getResult().setType(
          llvm::cast<mlir::ShapedType>(reshape->getResult(0).getType()));
    });
    rewriter.replaceOp(reshape, read.getResult());
    return llvm::success();
  }
};

/// Folds a `tensor.(collapse|expand)_shape` feeding a `ktdf.write_to_fifo`
/// into the write: a FIFO slot is flat, so the write may take the reshape's
/// source directly.
struct FoldReshapeIntoFifoWrite
    : mlir::OpRewritePattern<mlir::ktdf::WriteToFifoOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdf::WriteToFifoOp write,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto* const reshape = write.getData().getDefiningOp();
    if (!llvm::isa_and_present<mlir::tensor::CollapseShapeOp,
                               mlir::tensor::ExpandShapeOp>(reshape)) {
      return rewriter.notifyMatchFailure(write, "not written from a reshape");
    }

    rewriter.modifyOpInPlace(write, [&]() {
      write.getDataMutable().assign(reshape->getOperand(0));
    });
    return llvm::success();
  }
};

}  // namespace

void scheduler::populateFoldFifoReshapePatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<FoldReshapeOfFifoRead, FoldReshapeIntoFifoWrite>(
      patterns.getContext());
}

void scheduler::populateConvertToDataTransferPatterns(
    mlir::RewritePatternSet& patterns) {
  patterns.add<LowerLoadToFifo, LowerStoreFromFifo, LowerStoreFromLoad,
               LowerIndLoadToFifo, LowerIndStoreFromFifo>(
      patterns.getContext());
  populateFoldFifoReshapePatterns(patterns);
}
