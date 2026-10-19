//===-- KTIRBufferize.cpp ---------------------------------------*- c++ -*-===//
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <limits>

#include "Utils.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Transforms/Utils/IndirectAccessTileNarrowing.h"
#include "ktir/Dialect/KTDP/KTDP.h"

#define PASS_NAME "ktir-bufferize"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable KTIR Bufferize pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTIRBUFFERIZEPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return true when the access tile is defined by a
/// ConstructIndirectAccessTileOp (indirect tile).
[[nodiscard]] bool isIndirectTile(mlir::Value tile) {
  return tile.getDefiningOp<
             mlir::ktdp_lowering::ConstructIndirectAccessTileOp>() != nullptr;
}

/// Look up the static strides of a construct_memory_view-produced memref.
/// Returns the strides from the op's static_strides attribute when defined by
/// ktdp.construct_memory_view or ktdp_lowering.construct_memory_view;
/// otherwise falls back to the type's StridedLayoutAttr or row-major.
[[nodiscard]] llvm::SmallVector<int64_t> getMemRefStrides(MemRef memory_view) {
  mlir::MemRefType view_type = memory_view.getType();
  unsigned rank = view_type.getRank();

  // ktdp.construct_memory_view
  if (auto cmv =
          memory_view.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>()) {
    llvm::SmallVector<int64_t> strides(cmv.getStaticStrides().begin(),
                                       cmv.getStaticStrides().end());
    return strides;
  }
  // ktdp_lowering.construct_memory_view
  if (auto cmv =
          memory_view
              .getDefiningOp<mlir::ktdp_lowering::ConstructMemoryViewOp>()) {
    llvm::SmallVector<int64_t> strides(cmv.getStaticStrides().begin(),
                                       cmv.getStaticStrides().end());
    return strides;
  }
  // StridedLayoutAttr on the type.
  if (auto sl =
          mlir::dyn_cast<mlir::StridedLayoutAttr>(view_type.getLayout())) {
    return llvm::SmallVector<int64_t>(sl.getStrides().begin(),
                                      sl.getStrides().end());
  }
  // Fall back to row-major.
  llvm::SmallVector<int64_t> strides(rank);
  int64_t s = 1;
  for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
    strides[i] = s;
    s *= view_type.getShape()[i];
  }
  return strides;
}

/// Emit a `memref.memory_space_cast` of @p memory_view with the mapped memory
/// space. Returns the cast result (a memref with the same shape/layout but the
/// target memory space). Used by the direct load/store path.
[[nodiscard]] mlir::Value emitMemorySpaceCast(
    mlir::OpBuilder& builder, mlir::Location loc, MemRef memory_view,
    const AttrMapping& mem_space_map) {
  mlir::MemRefType view_type = memory_view.getType();
  mlir::Attribute memory_space = mem_space_map.map(getMemorySpace(memory_view));
  auto cast_type =
      mlir::MemRefType::get(view_type.getShape(), view_type.getElementType(),
                            view_type.getLayout(), memory_space);
  return mlir::memref::MemorySpaceCastOp::create(builder, loc, cast_type,
                                                 memory_view)
      .getResult();
}

/// Emit the base-view recipe for the IAB (indirect) path:
///   memory_space_cast + full-rank reinterpret_cast at offset 0 and given
///   sizes.  Strides come from the memref type layout.
///
/// For the IAB path the memory space is always remapped (emit_space_cast=true)
/// or already correct (emit_space_cast=false).
[[nodiscard]] mlir::Value buildBaseView(mlir::OpBuilder& builder,
                                        mlir::Location loc, MemRef memory_view,
                                        mlir::Value offset,
                                        llvm::ArrayRef<int64_t> sizes,
                                        const AttrMapping& mem_space_map,
                                        bool emit_space_cast) {
  mlir::MemRefType view_type = memory_view.getType();

  // Collect layout strides from the op if available, else from type/row-major.
  llvm::SmallVector<int64_t> strides = getMemRefStrides(memory_view);

  // Cast source memref value.
  mlir::Value cast_src = memory_view;
  mlir::Attribute memory_space;
  if (emit_space_cast) {
    memory_space = mem_space_map.map(getMemorySpace(memory_view));
    auto cast_type =
        mlir::MemRefType::get(view_type.getShape(), view_type.getElementType(),
                              view_type.getLayout(), memory_space);
    cast_src = mlir::memref::MemorySpaceCastOp::create(builder, loc, cast_type,
                                                       memory_view)
                   .getResult();
  } else {
    memory_space = view_type.getMemorySpace();
  }

  // Build result type: strided<[s0,s1,...], offset: ?> in memory_space.
  llvm::SmallVector<mlir::OpFoldResult> ofr_sizes, ofr_strides;
  for (int64_t sz : sizes) ofr_sizes.push_back(builder.getIndexAttr(sz));
  for (int64_t st : strides) ofr_strides.push_back(builder.getIndexAttr(st));

  const auto result_layout = mlir::StridedLayoutAttr::get(
      builder.getContext(), mlir::ShapedType::kDynamic, strides);
  const auto result_type = mlir::MemRefType::get(
      llvm::SmallVector<int64_t>(sizes.begin(), sizes.end()),
      view_type.getElementType(), result_layout, memory_space);

  return mlir::memref::ReinterpretCastOp::create(
             builder, loc, result_type, cast_src, mlir::OpFoldResult(offset),
             ofr_sizes, ofr_strides)
      .getResult();
}

/// The throttle of a store that fills an indirect address buffer: entries are
/// written one at a time.
constexpr int64_t kIndAddrBufFillThrottle = 1;

/// Determines whether @p access_tile is over an indirect address buffer, i.e.
/// over a `ktdp_lowering.construct_memory_view` (which carries the arch-graph
/// memory space of such a buffer).
[[nodiscard]] bool isIndAddrBufTile(mlir::Value access_tile) {
  auto tile = access_tile.getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
  return tile &&
         tile.getBase()
             .getDefiningOp<mlir::ktdp_lowering::ConstructMemoryViewOp>();
}

/// Gets the minimum throttle of the ops consuming @p value , looking through
/// `tensor.(collapse|expand)_shape`.
[[nodiscard]] int64_t getConsumerThrottle(mlir::Value value) {
  auto result = std::numeric_limits<int64_t>::max();
  for (auto* user : value.getUsers()) {
    result = std::min(
        result,
        mlir::isa<mlir::tensor::CollapseShapeOp, mlir::tensor::ExpandShapeOp>(
            user)
            ? getConsumerThrottle(user->getResult(0))
            : getThrottle(user));
  }
  return result;
}

/// Gets the throttle of the op producing @p value , looking through
/// `tensor.(collapse|expand)_shape`.
[[nodiscard]] int64_t getProducerThrottle(mlir::Value value) {
  while (auto* op = value.getDefiningOp()) {
    if (!mlir::isa<mlir::tensor::CollapseShapeOp, mlir::tensor::ExpandShapeOp>(
            op))
      return getThrottle(op);
    value = op->getOperand(0);
  }
  return std::numeric_limits<int64_t>::max();
}

/// Copies the discardable attributes of @p from onto @p to . If that leaves
/// @p to without a throttle, uses @p fallback , unless it is unknown.
void copyAttrsAndThrottle(mlir::Operation* from, mlir::Operation* to,
                          int64_t fallback) {
  to->setDiscardableAttrs(from->getRawDictionaryAttrs());
  if (!to->hasAttr(kThrottleAttrName) &&
      fallback != std::numeric_limits<int64_t>::max()) {
    setThrottle(to, fallback);
  }
}

/// Compute a flat-offset `Value` from per-dim indices and compile-time
/// strides. Optimises away zero-index terms and unit strides.
[[nodiscard]] mlir::Value computeReinterpretCastOffset(
    mlir::OpBuilder& builder, mlir::Location loc,
    llvm::ArrayRef<mlir::Value> indices, llvm::ArrayRef<int64_t> strides) {
  if (indices.empty())
    return mlir::arith::ConstantIndexOp::create(builder, loc, 0);

  llvm::SmallVector<mlir::Value> terms;
  for (auto [idx, stride] : llvm::zip(indices, strides)) {
    if (mlir::isZeroInteger(idx)) continue;
    mlir::Value term;
    if (stride == 1) {
      term = idx;
    } else {
      mlir::Value s =
          mlir::arith::ConstantIndexOp::create(builder, loc, stride);
      term = mlir::arith::MulIOp::create(builder, loc, idx, s);
    }
    terms.push_back(term);
  }
  if (terms.empty())
    return mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  if (terms.size() == 1) return terms[0];
  mlir::Value offset = terms[0];
  for (size_t i = 1; i < terms.size(); ++i)
    offset = mlir::arith::AddIOp::create(builder, loc, offset, terms[i]);
  return offset;
}

// ---------------------------------------------------------------------------
// Helpers shared by LowerLoad and LowerStore
// ---------------------------------------------------------------------------

/// Expand the `base_map` of @p tile_op over its indices and symbol operands,
/// returning one SSA Value per source-memref dimension.  Returns std::nullopt
/// and emits a match-failure diagnostic on @p anchor on error.
[[nodiscard]] std::optional<llvm::SmallVector<mlir::Value>> expandTileBaseMap(
    mlir::ktdp::ConstructAccessTilesOp tile_op, mlir::PatternRewriter& rewriter,
    mlir::Operation* anchor) {
  llvm::SmallVector<mlir::Value> expand_operands(tile_op.getIndices().begin(),
                                                 tile_op.getIndices().end());
  mlir::ValueRange symbol_operands = tile_op.getSymbolOperands();
  expand_operands.append(symbol_operands.begin(), symbol_operands.end());

  auto per_dim = mlir::affine::expandAffineMap(
      rewriter, tile_op.getLoc(), tile_op.getBaseMap(), expand_operands);
  if (!per_dim) {
    (void)rewriter.notifyMatchFailure(anchor, "unable to expand base_map");
    return std::nullopt;
  }
  return llvm::SmallVector<mlir::Value>(per_dim->begin(), per_dim->end());
}

// ---------------------------------------------------------------------------
// LowerLoad  (ktdp.load -> ktdp_lowering.load on a cast memref with
//             per-dim offsets, skips indirect tiles)
// ---------------------------------------------------------------------------

struct LowerLoad : mlir::OpRewritePattern<mlir::ktdp::LoadOp> {
  explicit LowerLoad(mlir::MLIRContext* context,
                     const AttrMapping& mem_space_map)
      : OpRewritePattern(context), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(mlir::ktdp::LoadOp load,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    // Skip indirect tiles — handled by LowerIndirectLoad.
    if (isIndirectTile(load.getAccessTile()))
      return rewriter.notifyMatchFailure(load, "indirect tile");

    const auto result_type =
        mlir::dyn_cast<mlir::RankedTensorType>(load.getType());
    if (!result_type || !result_type.hasStaticShape())
      return rewriter.notifyMatchFailure(load,
                                         "not a static-shape ranked tensor");

    auto tile_op = load.getAccessTile()
                       .getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
    if (!tile_op)
      return rewriter.notifyMatchFailure(load,
                                         "access tile not locally defined");

    const auto memory_view = llvm::dyn_cast<MemRef>(tile_op.getBase());
    if (!memory_view)
      return rewriter.notifyMatchFailure(load, "base is not a memref");

    // Expand base_map to get one index per source-memref dimension.
    auto indices = expandTileBaseMap(tile_op, rewriter, load);
    if (!indices) return llvm::failure();

    // Emit only the memory-space cast; address offsets go on the load op.
    // Skip the cast for the IAB fill path (already carries correct mem space).
    mlir::Value source = memory_view;
    bool is_iab_fill_path =
        memory_view
            .getDefiningOp<mlir::ktdp_lowering::ConstructMemoryViewOp>() !=
        nullptr;
    if (!is_iab_fill_path)
      source = emitMemorySpaceCast(rewriter, load.getLoc(), memory_view,
                                   mem_space_map_);

    // Build mixed offsets (per-dim indices) and unit strides.
    llvm::SmallVector<mlir::OpFoldResult> mixed_offsets, mixed_sizes,
        mixed_strides;
    for (mlir::Value idx : *indices) mixed_offsets.push_back(idx);
    for (int64_t sz : result_type.getShape())
      mixed_sizes.push_back(rewriter.getIndexAttr(sz));
    mixed_strides.assign(result_type.getRank(), rewriter.getIndexAttr(1));

    auto new_load = mlir::ktdp_lowering::LoadOp::create(
        rewriter, load.getLoc(), result_type, source, mixed_offsets,
        mixed_sizes, mixed_strides);
    // Like a tiled load, transfer at the rate of the consuming computes.
    copyAttrsAndThrottle(load, new_load, getConsumerThrottle(load.getResult()));
    rewriter.replaceOp(load, new_load);

    // Erase the tile op if it has no remaining users.
    if (tile_op.getResult().use_empty()) rewriter.eraseOp(tile_op);
    return llvm::success();
  }

 private:
  AttrMapping mem_space_map_;
};

// ---------------------------------------------------------------------------
// LowerStore  (ktdp.store -> ktdp_lowering.store on a cast memref with
//              per-dim offsets, skips indirect tiles)
// ---------------------------------------------------------------------------

struct LowerStore : mlir::OpRewritePattern<mlir::ktdp::StoreOp> {
  explicit LowerStore(mlir::MLIRContext* context,
                      const AttrMapping& mem_space_map)
      : OpRewritePattern(context), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(mlir::ktdp::StoreOp store,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    // Skip indirect tiles — handled by LowerIndirectStore.
    if (isIndirectTile(store.getAccessTile()))
      return rewriter.notifyMatchFailure(store, "indirect tile");

    const auto data_type = store.getDataTile().getType();
    if (!data_type.hasStaticShape())
      return rewriter.notifyMatchFailure(store, "dynamic shape");

    auto tile_op = store.getAccessTile()
                       .getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
    if (!tile_op)
      return rewriter.notifyMatchFailure(store,
                                         "access tile not locally defined");

    const auto memory_view = llvm::dyn_cast<MemRef>(tile_op.getBase());
    if (!memory_view)
      return rewriter.notifyMatchFailure(store, "base is not a memref");

    // Expand base_map to get one index per source-memref dimension.
    auto indices = expandTileBaseMap(tile_op, rewriter, store);
    if (!indices) return llvm::failure();

    // Emit only the memory-space cast; address offsets go on the store op.
    // Skip the cast for the IAB fill path (already carries correct mem space).
    mlir::Value dest = memory_view;
    bool is_iab_fill_path =
        memory_view
            .getDefiningOp<mlir::ktdp_lowering::ConstructMemoryViewOp>() !=
        nullptr;
    if (!is_iab_fill_path)
      dest = emitMemorySpaceCast(rewriter, store.getLoc(), memory_view,
                                 mem_space_map_);

    // Build mixed offsets (per-dim indices) and unit strides.
    llvm::SmallVector<mlir::OpFoldResult> mixed_offsets, mixed_sizes,
        mixed_strides;
    for (mlir::Value idx : *indices) mixed_offsets.push_back(idx);
    for (int64_t sz : data_type.getShape())
      mixed_sizes.push_back(rewriter.getIndexAttr(sz));
    mixed_strides.assign(data_type.getRank(), rewriter.getIndexAttr(1));

    auto new_store = mlir::ktdp_lowering::StoreOp::create(
        rewriter, store.getLoc(), store.getDataTile(), dest, mixed_offsets,
        mixed_sizes, mixed_strides);
    // Like a tiled store, transfer at the rate of the producing compute.
    copyAttrsAndThrottle(store, new_store,
                         isIndAddrBufTile(store.getAccessTile())
                             ? kIndAddrBufFillThrottle
                             : getProducerThrottle(store.getDataTile()));
    rewriter.eraseOp(store);

    // Erase the tile op if it has no remaining users.
    if (tile_op.getResult().use_empty()) rewriter.eraseOp(tile_op);
    return llvm::success();
  }

 private:
  AttrMapping mem_space_map_;
};

// ---------------------------------------------------------------------------
// LowerTiledLoad / LowerTiledStore
//   ktdp_lowering.load/store whose source/dest is a direct access tile
//   (produced by KTIRMapAndTile) -> replace the tile with a cast memref.
//   The existing offsets/sizes/strides on the op already encode the slice
//   within the tile; we only need to swap out the source/dest operand.
// ---------------------------------------------------------------------------

struct LowerTiledLoad : mlir::OpRewritePattern<mlir::ktdp_lowering::LoadOp> {
  explicit LowerTiledLoad(mlir::MLIRContext* context,
                          const AttrMapping& mem_space_map)
      : OpRewritePattern(context), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(mlir::ktdp_lowering::LoadOp load,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    // Only handle loads whose source is a direct (non-indirect) access tile.
    if (!mlir::isa<mlir::ktdp::AccessTileType>(load.getSource().getType()))
      return rewriter.notifyMatchFailure(load, "source is not an access tile");
    if (isIndirectTile(load.getSource()))
      return rewriter.notifyMatchFailure(load, "indirect tile");

    auto tile_op =
        load.getSource().getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
    if (!tile_op)
      return rewriter.notifyMatchFailure(load,
                                         "access tile not locally defined");

    const auto memory_view = llvm::dyn_cast<MemRef>(tile_op.getBase());
    if (!memory_view)
      return rewriter.notifyMatchFailure(load, "base is not a memref");

    auto indices = expandTileBaseMap(tile_op, rewriter, load);
    if (!indices) return llvm::failure();

    bool has_non_zero_offset = llvm::any_of(
        *indices, [](mlir::Value v) { return !mlir::isZeroInteger(v); });

    bool is_iab_fill_path =
        memory_view
            .getDefiningOp<mlir::ktdp_lowering::ConstructMemoryViewOp>() !=
        nullptr;

    mlir::Value source;
    if (has_non_zero_offset) {
      llvm::SmallVector<int64_t> strides = getMemRefStrides(memory_view);
      mlir::Value offset = computeReinterpretCastOffset(rewriter, load.getLoc(),
                                                        *indices, strides);
      llvm::ArrayRef<int64_t> tile_shape = tile_op.getType().getShape();
      source = buildBaseView(rewriter, load.getLoc(), memory_view, offset,
                             tile_shape, mem_space_map_,
                             /*emit_space_cast=*/!is_iab_fill_path);
    } else {
      source = is_iab_fill_path
                   ? mlir::Value(memory_view)
                   : emitMemorySpaceCast(rewriter, load.getLoc(), memory_view,
                                         mem_space_map_);
    }

    // Rebuild the load with the cast memref as source, keeping the slice.
    auto new_load = mlir::ktdp_lowering::LoadOp::create(
        rewriter, load.getLoc(), load.getType(), source,
        mlir::cast<mlir::OffsetSizeAndStrideOpInterface>(load.getOperation()));
    new_load->setDiscardableAttrs(load->getRawDictionaryAttrs());
    rewriter.replaceOp(load, new_load);

    // Erase the tile op if it has no remaining users.
    if (tile_op.getResult().use_empty()) rewriter.eraseOp(tile_op);
    return llvm::success();
  }

 private:
  AttrMapping mem_space_map_;
};

struct LowerTiledStore : mlir::OpRewritePattern<mlir::ktdp_lowering::StoreOp> {
  explicit LowerTiledStore(mlir::MLIRContext* context,
                           const AttrMapping& mem_space_map)
      : OpRewritePattern(context), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(mlir::ktdp_lowering::StoreOp store,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    // Only handle stores whose dest is a direct (non-indirect) access tile.
    if (!mlir::isa<mlir::ktdp::AccessTileType>(store.getDest().getType()))
      return rewriter.notifyMatchFailure(store, "dest is not an access tile");
    if (isIndirectTile(store.getDest()))
      return rewriter.notifyMatchFailure(store, "indirect tile");

    auto tile_op =
        store.getDest().getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
    if (!tile_op)
      return rewriter.notifyMatchFailure(store,
                                         "access tile not locally defined");

    const auto memory_view = llvm::dyn_cast<MemRef>(tile_op.getBase());
    if (!memory_view)
      return rewriter.notifyMatchFailure(store, "base is not a memref");

    auto indices = expandTileBaseMap(tile_op, rewriter, store);
    if (!indices) return llvm::failure();

    bool has_non_zero_offset = llvm::any_of(
        *indices, [](mlir::Value v) { return !mlir::isZeroInteger(v); });

    bool is_iab_fill_path =
        memory_view
            .getDefiningOp<mlir::ktdp_lowering::ConstructMemoryViewOp>() !=
        nullptr;

    mlir::Value dest;
    if (has_non_zero_offset) {
      llvm::SmallVector<int64_t> strides = getMemRefStrides(memory_view);
      mlir::Value offset = computeReinterpretCastOffset(
          rewriter, store.getLoc(), *indices, strides);
      llvm::ArrayRef<int64_t> tile_shape = tile_op.getType().getShape();
      dest = buildBaseView(rewriter, store.getLoc(), memory_view, offset,
                           tile_shape, mem_space_map_,
                           /*emit_space_cast=*/!is_iab_fill_path);
    } else {
      dest = is_iab_fill_path
                 ? mlir::Value(memory_view)
                 : emitMemorySpaceCast(rewriter, store.getLoc(), memory_view,
                                       mem_space_map_);
    }

    // Rebuild the store with the cast memref as dest, keeping the slice.
    auto new_store = mlir::ktdp_lowering::StoreOp::create(
        rewriter, store.getLoc(), store.getSource(), dest,
        mlir::cast<mlir::OffsetSizeAndStrideOpInterface>(store.getOperation()));
    new_store->setDiscardableAttrs(store->getRawDictionaryAttrs());
    rewriter.eraseOp(store);

    // Erase the tile op if it has no remaining users.
    if (tile_op.getResult().use_empty()) rewriter.eraseOp(tile_op);
    return llvm::success();
  }

 private:
  AttrMapping mem_space_map_;
};

// ---------------------------------------------------------------------------
// Helpers for the indirect patterns
// ---------------------------------------------------------------------------

/// Given a ConstructIndirectAccessTileOp and an optional slice (from
/// ktdp_lowering::LoadOp / StoreOp), compute the mixed offsets/sizes/strides
/// per dimension of `base` needed for ind_load / ind_store.
///
/// Returns false on failure (diagnostic already emitted on `tile_op`).
///
/// The function also fills:
///   `iab_index_value`  — the SSA Value to use as ind_addr_buf_index
///   `iab_memref_value` — the raw (un-reinterpreted) IAB memref from the op
///
struct IndirectOpInfo {
  llvm::SmallVector<mlir::OpFoldResult> offsets;
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  llvm::SmallVector<mlir::OpFoldResult> strides;
  mlir::Value iab_index;
  mlir::RankedTensorType tensor_type;
};

[[nodiscard]] bool deriveIndirectOpInfo(
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp tile_op,
    mlir::RankedTensorType tensor_type,
    // Optional slice from ktdp_lowering::LoadOp / StoreOp after map-and-tile.
    // Pass empty if derived from ktdp::LoadOp / ktdp::StoreOp (full tile).
    std::optional<mlir::OffsetSizeAndStrideOpInterface> slice_iface,
    IndirectOpInfo& info) {
  mlir::MLIRContext* ctx = tile_op.getContext();
  mlir::Location loc = tile_op.getLoc();

  auto positions = tile_op.getIndAddrBufDimPositions();
  auto per_dim_maps = tile_op.getPerDimSubscriptMaps();
  auto captured = tile_op.getCapturedVariables();
  auto ivs = tile_op.getIntermediateVariables();
  mlir::IntegerSet vars_space_set = tile_op.getVariablesSpaceSet().getValue();
  mlir::AffineMap vars_space_order = tile_op.getVariablesSpaceOrder();

  unsigned num_captured = static_cast<unsigned>(captured.size());
  unsigned num_ivs = static_cast<unsigned>(ivs.size());
  unsigned base_rank = static_cast<unsigned>(per_dim_maps.size());
  unsigned unified_dims = num_captured + num_ivs;

  // Step 1: ind_addr_buf_dim_positions must have exactly 1 entry.
  if (positions.size() != 1) {
    tile_op.emitError(
        "LowerIndirectLoad/Store: ind_addr_buf_dim_positions must have size 1");
    return false;
  }
  unsigned iab_pos = static_cast<unsigned>(positions[0]);

  // Build unified variable list: [captured..., ivs...]
  llvm::SmallVector<mlir::Value> all_vars(captured.begin(), captured.end());
  for (mlir::Value iv : ivs) all_vars.push_back(iv);

  // Determine the IAB index value.
  if (iab_pos < num_captured) {
    // Captured variable — use its SSA value directly.
    info.iab_index = all_vars[iab_pos];
  } else {
    // Intermediate variable (iv).  Its extent must be 1 in vars_space_set.
    unsigned iv_idx = iab_pos - num_captured;
    auto trip = getTripCount(vars_space_set, iv_idx);
    if (!trip || *trip != 1) {
      tile_op.emitError(
          "LowerIndirectLoad/Store: IAB-indexed iv does not have extent 1");
      return false;
    }
    // Extent 1 — use constant 0.
    mlir::OpBuilder builder(tile_op);
    info.iab_index =
        mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  }

  // Step 2: Decompose per_dim_subscript_maps[d] into:
  //   - captured_part[d]: an AffineMap over just the captured dims
  //   - dim_uses_iv[d]:   which iv (0..num_ivs-1) the dim uses, if any
  // Each iv may appear in at most one base dim.
  llvm::SmallVector<std::optional<unsigned>> iv_to_base_dim(num_ivs,
                                                            std::nullopt);
  llvm::SmallVector<mlir::AffineMap> captured_part(base_rank);
  llvm::SmallVector<std::optional<unsigned>> dim_uses_iv(base_rank,
                                                         std::nullopt);

  // Helper: build a dim-replacement array that zeros out all iv dims and
  // remaps captured dims to 0..num_captured-1 (identity).
  auto make_captured_repls = [&]() {
    llvm::SmallVector<mlir::AffineExpr> repls(unified_dims);
    for (unsigned i = 0; i < unified_dims; ++i) {
      if (i < num_captured)
        repls[i] = mlir::getAffineDimExpr(i, ctx);
      else
        repls[i] = mlir::getAffineConstantExpr(0, ctx);
    }
    return repls;
  };
  const auto captured_repls = make_captured_repls();

  for (unsigned d = 0; d < base_rank; ++d) {
    auto map_attr = mlir::dyn_cast<mlir::AffineMapAttr>(per_dim_maps[d]);
    if (!map_attr) {
      tile_op.emitError(
          "LowerIndirectLoad/Store: per_dim_subscript_maps element is not an "
          "AffineMapAttr");
      return false;
    }
    mlir::AffineMap m = map_attr.getValue();
    assert(m.getNumResults() == 1 && "per-dim map should have 1 result");
    mlir::AffineExpr expr = m.getResult(0);

    // For each iv k, try: candidate = simplify(expr - d_{num_captured+k}).
    // If the candidate references no iv dim, then expr = f_d(captured) + iv_k.
    std::optional<unsigned> found_iv;
    for (unsigned k = 0; k < num_ivs; ++k) {
      unsigned dim = num_captured + k;
      mlir::AffineExpr iv_expr = mlir::getAffineDimExpr(dim, ctx);
      mlir::AffineExpr candidate =
          mlir::simplifyAffineExpr(expr - iv_expr, unified_dims, 0);

      bool refs_iv = false;
      candidate.walk([&](mlir::AffineExpr e) {
        if (auto de = mlir::dyn_cast<mlir::AffineDimExpr>(e))
          if (de.getPosition() >= num_captured) refs_iv = true;
      });

      if (!refs_iv) {
        found_iv = k;
        // Build the captured-only part: substitute iv dims with 0.
        mlir::AffineExpr cap_expr =
            candidate.replaceDimsAndSymbols(captured_repls, {});
        captured_part[d] =
            mlir::AffineMap::get(num_captured, 0, {cap_expr}, ctx);
        break;
      }
    }
    if (!found_iv) {
      // No iv — f_d(captured) only.
      mlir::AffineExpr cap_expr =
          expr.replaceDimsAndSymbols(captured_repls, {});
      captured_part[d] = mlir::AffineMap::get(num_captured, 0, {cap_expr}, ctx);
      dim_uses_iv[d] = std::nullopt;
    } else {
      unsigned k = *found_iv;
      if (iv_to_base_dim[k].has_value()) {
        tile_op.emitError(
            "LowerIndirectLoad/Store: iv appears in more than one base dim");
        return false;
      }
      iv_to_base_dim[k] = d;
      dim_uses_iv[d] = k;
    }
  }

  // Step 3: vars_space_order maps iv index -> tile dimension.
  // vars_space_order is a permutation map from iv-space to tile-space.
  // tile_dim t(k) = vars_space_order.getResult(k) evaluated at (0..k..0) =
  //   the constant or dim result at position k.
  // Since vars_space_order is typically affine_map<(d0,...,dN-1) ->
  // (d0,...,dN-1)> we extract the output index for each input. For the common
  // permutation: result[k] = affine_dim_expr(perm(k)). We use a helper:
  // evaluate result[k] symbolically.
  auto get_tile_dim = [&](unsigned k) -> std::optional<unsigned> {
    if (k >= vars_space_order.getNumResults()) return std::nullopt;
    mlir::AffineExpr r = vars_space_order.getResult(k);
    if (auto d = mlir::dyn_cast<mlir::AffineDimExpr>(r)) return d.getPosition();
    // Constant expression (shouldn't happen for a valid order map)
    return std::nullopt;
  };

  // Step 4: Build slice offsets/sizes/strides per base dim.
  // If we have a slice (from ktdp_lowering::Load/StoreOp), use its mixed
  // offsets/sizes/strides per tile dimension.  Otherwise the slice is the
  // full tile: offset=0, size=N_k, stride=1.

  // Get trip counts from vars_space_set for each iv.
  llvm::SmallVector<int64_t> trip_counts(num_ivs, 1);
  for (unsigned k = 0; k < num_ivs; ++k) {
    auto tc = getTripCount(vars_space_set, k);
    if (tc) trip_counts[k] = *tc;
  }

  // Retrieve slice info (mixed offsets/sizes/strides) indexed by tile dim.
  llvm::SmallVector<mlir::OpFoldResult> s_offsets, s_sizes, s_strides;
  if (slice_iface) {
    s_offsets = slice_iface->getMixedOffsets();
    s_sizes = slice_iface->getMixedSizes();
    s_strides = slice_iface->getMixedStrides();
  }

  // Materialize f_d(captured) using affine.apply.
  mlir::OpBuilder builder(tile_op);
  llvm::SmallVector<mlir::Value> captured_vals(captured.begin(),
                                               captured.end());

  for (unsigned d = 0; d < base_rank; ++d) {
    if (dim_uses_iv[d].has_value()) {
      unsigned k = *dim_uses_iv[d];
      auto tile_dim_opt = get_tile_dim(k);
      if (!tile_dim_opt.has_value()) {
        tile_op.emitError(
            "LowerIndirectLoad/Store: unable to determine tile dim for iv");
        return false;
      }
      unsigned tile_dim = *tile_dim_opt;

      // f_d(captured): materialized via affine.apply if non-trivial.
      mlir::OpFoldResult f_d;
      if (captured_vals.empty() ||
          captured_part[d].getResult(0).isSymbolicOrConstant()) {
        // Constant expression.
        if (auto c = mlir::getConstantIntValue(mlir::OpFoldResult(
                mlir::AffineMapAttr::get(captured_part[d])))) {
          f_d = builder.getIndexAttr(*c);
        } else {
          // Expand affine map over captured values.
          auto expanded = mlir::affine::expandAffineMap(
              builder, loc, captured_part[d], captured_vals);
          if (!expanded || expanded->empty()) {
            tile_op.emitError(
                "LowerIndirectLoad/Store: failed to expand captured-part map");
            return false;
          }
          f_d = (*expanded)[0];
        }
      } else {
        auto expanded = mlir::affine::expandAffineMap(
            builder, loc, captured_part[d], captured_vals);
        if (!expanded || expanded->empty()) {
          tile_op.emitError(
              "LowerIndirectLoad/Store: failed to expand captured-part map");
          return false;
        }
        f_d = (*expanded)[0];
      }

      // offset_d = f_d + o[tile_dim]
      mlir::OpFoldResult o_td;
      if (slice_iface) {
        o_td = s_offsets[tile_dim];
      } else {
        o_td = builder.getIndexAttr(0);
      }

      mlir::OpFoldResult offset_d;
      auto f_d_const = mlir::getConstantIntValue(f_d);
      auto o_const = mlir::getConstantIntValue(o_td);
      if (f_d_const && o_const) {
        offset_d = builder.getIndexAttr(*f_d_const + *o_const);
      } else if (f_d_const && *f_d_const == 0) {
        offset_d = o_td;
      } else if (o_const && *o_const == 0) {
        offset_d = f_d;
      } else {
        // Dynamic — emit arith.addi.
        auto f_val = mlir::getValueOrCreateConstantIndexOp(builder, loc, f_d);
        auto o_val = mlir::getValueOrCreateConstantIndexOp(builder, loc, o_td);
        offset_d =
            mlir::arith::AddIOp::create(builder, loc, f_val, o_val).getResult();
      }
      info.offsets.push_back(offset_d);

      // size_d = s[tile_dim]
      if (slice_iface) {
        info.sizes.push_back(s_sizes[tile_dim]);
      } else {
        info.sizes.push_back(builder.getIndexAttr(trip_counts[k]));
      }

      // stride_d = t[tile_dim] (always static 1 in practice; verifier enforces)
      if (slice_iface) {
        info.strides.push_back(s_strides[tile_dim]);
      } else {
        info.strides.push_back(builder.getIndexAttr(1));
      }
    } else {
      // Dim d has no iv — pure captured offset.
      // f_d(captured):
      mlir::OpFoldResult f_d;
      if (captured_vals.empty() ||
          captured_part[d].getResult(0).isSymbolicOrConstant()) {
        if (auto c = mlir::getConstantIntValue(mlir::OpFoldResult(
                mlir::AffineMapAttr::get(captured_part[d])))) {
          f_d = builder.getIndexAttr(*c);
        } else {
          auto expanded = mlir::affine::expandAffineMap(
              builder, loc, captured_part[d], captured_vals);
          if (!expanded || expanded->empty()) {
            tile_op.emitError(
                "LowerIndirectLoad/Store: failed to expand captured-part map");
            return false;
          }
          f_d = (*expanded)[0];
        }
      } else {
        auto expanded = mlir::affine::expandAffineMap(
            builder, loc, captured_part[d], captured_vals);
        if (!expanded || expanded->empty()) {
          tile_op.emitError(
              "LowerIndirectLoad/Store: failed to expand captured-part map");
          return false;
        }
        f_d = (*expanded)[0];
      }
      info.offsets.push_back(f_d);
      info.sizes.push_back(builder.getIndexAttr(1));
      info.strides.push_back(builder.getIndexAttr(1));
    }
  }

  // Fold constant offsets (e.g. a captured `arith.constant`) into static ones.
  for (auto& offset : info.offsets) {
    if (auto value = llvm::dyn_cast<mlir::Value>(offset))
      offset = mlir::getAsOpFoldResult(value);
  }

  info.tensor_type = tensor_type;
  return true;
}

/// Build the IAB memref view: reinterpret_cast to strided<[1], offset: ?>,
/// keeping the existing memory space (no memory_space_cast).
[[nodiscard]] mlir::Value buildIABView(mlir::OpBuilder& builder,
                                       mlir::Location loc,
                                       mlir::Value iab_memref) {
  auto iab_type = mlir::cast<mlir::MemRefType>(iab_memref.getType());
  // Result type: memref<NxElem, strided<[1], offset: ?>, mem_space>
  const auto strided = mlir::StridedLayoutAttr::get(
      builder.getContext(), mlir::ShapedType::kDynamic, {1});
  auto result_type =
      mlir::MemRefType::get(iab_type.getShape(), iab_type.getElementType(),
                            strided, iab_type.getMemorySpace());

  // If already strided<[1], offset: ?> with same memory space, no cast needed.
  if (iab_type == result_type) return iab_memref;

  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  llvm::SmallVector<mlir::OpFoldResult> sizes = {
      builder.getIndexAttr(iab_type.getDimSize(0))};
  llvm::SmallVector<mlir::OpFoldResult> strides = {builder.getIndexAttr(1)};
  return mlir::memref::ReinterpretCastOp::create(
             builder, loc, result_type, iab_memref, mlir::OpFoldResult(c0),
             sizes, strides)
      .getResult();
}

// ---------------------------------------------------------------------------
// LowerIndirectLoad  (ktdp.load or ktdp_lowering.load on an indirect tile
//                     -> ktdp_lowering.ind_load)
// ---------------------------------------------------------------------------

template <typename LoadOpT>
struct LowerIndirectLoad : mlir::OpRewritePattern<LoadOpT> {
  explicit LowerIndirectLoad(mlir::MLIRContext* ctx,
                             const AttrMapping& mem_space_map)
      : mlir::OpRewritePattern<LoadOpT>(ctx), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(LoadOpT load, mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    // Identify the access tile.
    mlir::Value tile_val;
    if constexpr (std::is_same_v<LoadOpT, mlir::ktdp::LoadOp>) {
      tile_val = load.getAccessTile();
    } else {
      // ktdp_lowering::LoadOp — source must be an access tile.
      tile_val = load.getSource();
      if (!mlir::isa<mlir::ktdp::AccessTileType>(tile_val.getType()))
        return rewriter.notifyMatchFailure(load,
                                           "source is not an access tile");
    }

    auto tile_op = tile_val.getDefiningOp<
        mlir::ktdp_lowering::ConstructIndirectAccessTileOp>();
    if (!tile_op)
      return rewriter.notifyMatchFailure(load, "not an indirect tile");

    auto result_type =
        mlir::dyn_cast<mlir::RankedTensorType>(load.getResult().getType());
    if (!result_type)
      return rewriter.notifyMatchFailure(load, "result is not a ranked tensor");

    // Optionally extract the slice interface (ktdp_lowering::LoadOp path).
    std::optional<mlir::OffsetSizeAndStrideOpInterface> slice_iface;
    if constexpr (std::is_same_v<LoadOpT, mlir::ktdp_lowering::LoadOp>) {
      slice_iface =
          mlir::cast<mlir::OffsetSizeAndStrideOpInterface>(load.getOperation());
    }

    IndirectOpInfo info;
    if (!deriveIndirectOpInfo(tile_op, result_type, slice_iface, info))
      return llvm::failure();

    mlir::Location loc = load.getLoc();

    // Build base view: memory_space_cast + full-rank reinterpret_cast.
    auto base = mlir::cast<MemRef>(tile_op.getBase());
    auto base_type = base.getType();
    llvm::SmallVector<int64_t> full_sizes(base_type.getShape().begin(),
                                          base_type.getShape().end());

    mlir::Value c0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    mlir::Value base_view =
        buildBaseView(rewriter, loc, base, c0, full_sizes, mem_space_map_,
                      /*emit_space_cast=*/true);

    // Build IAB view.
    mlir::Value iab_view =
        buildIABView(rewriter, loc, tile_op.getIndAddrBufMemref());

    // Create ind_load.
    auto ind_load = mlir::ktdp_lowering::IndLoadOp::create(
        rewriter, loc, result_type, iab_view, info.iab_index, base_view,
        info.offsets, info.sizes, info.strides);
    // Like a direct load, transfer at the rate of the consuming computes.
    copyAttrsAndThrottle(load, ind_load,
                         getConsumerThrottle(load->getResult(0)));

    rewriter.replaceOp(load, ind_load.getResult());

    // Erase the tile op if it has no remaining users.
    if (tile_op.getResult().use_empty()) rewriter.eraseOp(tile_op);

    return llvm::success();
  }

 private:
  AttrMapping mem_space_map_;
};

// ---------------------------------------------------------------------------
// LowerIndirectStore  (ktdp.store or ktdp_lowering.store on an indirect tile
//                      -> ktdp_lowering.ind_store)
// ---------------------------------------------------------------------------

template <typename StoreOpT>
struct LowerIndirectStore : mlir::OpRewritePattern<StoreOpT> {
  explicit LowerIndirectStore(mlir::MLIRContext* ctx,
                              const AttrMapping& mem_space_map)
      : mlir::OpRewritePattern<StoreOpT>(ctx), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(StoreOpT store, mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    // Identify the access tile (dest).
    mlir::Value tile_val;
    mlir::Value source_val;
    if constexpr (std::is_same_v<StoreOpT, mlir::ktdp::StoreOp>) {
      tile_val = store.getAccessTile();
      source_val = store.getDataTile();
    } else {
      // ktdp_lowering::StoreOp
      tile_val = store.getDest();
      if (!mlir::isa<mlir::ktdp::AccessTileType>(tile_val.getType()))
        return rewriter.notifyMatchFailure(store, "dest is not an access tile");
      source_val = store.getSource();
    }

    auto tile_op = tile_val.getDefiningOp<
        mlir::ktdp_lowering::ConstructIndirectAccessTileOp>();
    if (!tile_op)
      return rewriter.notifyMatchFailure(store, "not an indirect tile");

    auto source_type =
        mlir::dyn_cast<mlir::RankedTensorType>(source_val.getType());
    if (!source_type)
      return rewriter.notifyMatchFailure(store,
                                         "source is not a ranked tensor");

    // Optionally extract the slice interface (ktdp_lowering::StoreOp path).
    std::optional<mlir::OffsetSizeAndStrideOpInterface> slice_iface;
    if constexpr (std::is_same_v<StoreOpT, mlir::ktdp_lowering::StoreOp>) {
      slice_iface = mlir::cast<mlir::OffsetSizeAndStrideOpInterface>(
          store.getOperation());
    }

    IndirectOpInfo info;
    if (!deriveIndirectOpInfo(tile_op, source_type, slice_iface, info))
      return llvm::failure();

    mlir::Location loc = store.getLoc();

    // Build base view: memory_space_cast + full-rank reinterpret_cast.
    auto base = mlir::cast<MemRef>(tile_op.getBase());
    auto base_type = base.getType();
    llvm::SmallVector<int64_t> full_sizes(base_type.getShape().begin(),
                                          base_type.getShape().end());

    mlir::Value c0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    mlir::Value base_view =
        buildBaseView(rewriter, loc, base, c0, full_sizes, mem_space_map_,
                      /*emit_space_cast=*/true);

    // Build IAB view.
    mlir::Value iab_view =
        buildIABView(rewriter, loc, tile_op.getIndAddrBufMemref());

    // Create ind_store.
    auto ind_store = mlir::ktdp_lowering::IndStoreOp::create(
        rewriter, loc, source_val, iab_view, info.iab_index, base_view,
        info.offsets, info.sizes, info.strides);
    // Like a direct store, transfer at the rate of the producing compute.
    copyAttrsAndThrottle(store, ind_store, getProducerThrottle(source_val));

    rewriter.eraseOp(store);

    // Erase the tile op if it has no remaining users.
    if (tile_op.getResult().use_empty()) rewriter.eraseOp(tile_op);

    return llvm::success();
  }

 private:
  AttrMapping mem_space_map_;
};

struct KTIRBufferizePass
    : public impl::KTIRBufferizePassBase<KTIRBufferizePass> {
  using KTIRBufferizePassBase<KTIRBufferizePass>::KTIRBufferizePassBase;
  void runOnOperation() override;
};

}  // namespace

void KTIRBufferizePass::runOnOperation() {
  if (disable_this_pass) {
    return;
  }

  mlir::func::FuncOp func = getOperation();
  // Obtain the default device and start a mapping.
  auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
  if (!default_device) {
    signalPassFailure();
    return;
  }

  // This pass supports remapping of the memory space attributes.
  AttrMapping mem_space_map;
  if (auto map = default_device->getAttrOfType<mlir::ktdf_arch::MapAttr>(
          "mem_space_mapping");
      map) {
    mem_space_map.insert_range(map);
  }

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<LowerLoad, LowerStore>(patterns.getContext(), mem_space_map);
  patterns.add<LowerTiledLoad, LowerTiledStore>(patterns.getContext(),
                                                mem_space_map);
  patterns.add<LowerIndirectLoad<mlir::ktdp::LoadOp>>(patterns.getContext(),
                                                      mem_space_map);
  patterns.add<LowerIndirectLoad<mlir::ktdp_lowering::LoadOp>>(
      patterns.getContext(), mem_space_map);
  patterns.add<LowerIndirectStore<mlir::ktdp::StoreOp>>(patterns.getContext(),
                                                        mem_space_map);
  patterns.add<LowerIndirectStore<mlir::ktdp_lowering::StoreOp>>(
      patterns.getContext(), mem_space_map);
  populateConvertToDataTransferPatterns(patterns);

  if (failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}
