//===-- KTIRMapAndTile.cpp --------------------------------------*- c++ -*-===//
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
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Transforms.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <cstdint>

#include "Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"

#define PASS_NAME "ktir-map-and-tile"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable KTIR Map and Tile pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTIRMAPANDTILEPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

const auto kSkipRegions = mlir::OpPrintingFlags().skipRegions();

using I64Vec = llvm::SmallVector<int64_t>;

struct KTIRMapAndTilePass
    : public impl::KTIRMapAndTilePassBase<KTIRMapAndTilePass> {
  using KTIRMapAndTilePassBase<KTIRMapAndTilePass>::KTIRMapAndTilePassBase;

  void runOnOperation() override;
};

/// Given a shape and a target vector_length, compute per-dimension tile sizes
/// working rightmost-first until the product of covered dims reaches
/// vector_length.  Uncovered dims (leftmost) get tile size 1 so a loop is
/// generated for them; covered dims get the largest value that divides both
/// the dim size and the remaining vector-length product.
[[nodiscard]] auto determineTileSizes(
    llvm::ArrayRef<int64_t> shape, int64_t vector_length,
    llvm::SmallVectorImpl<int64_t>& tile_sizes) -> llvm::LogicalResult {
  const int64_t rank = shape.size();

  // Find how many rightmost dims are needed to cover vector_length.
  int64_t product = 1;
  unsigned covered_dims = 0;
  while (covered_dims < shape.size()) {
    const auto size = shape[shape.size() - covered_dims - 1];
    if (mlir::ShapedType::isDynamic(size)) {
      // There weren't enough statically known dims to cover the vector size.
      return llvm::failure();
    }

    product *= size;
    covered_dims++;
    if (product >= vector_length) {
      break;
    }
  }

  // Uncovered dims (leftmost) get tile size 1 — a loop is generated that
  // iterates over the full extent one element at a time.
  tile_sizes.resize(rank, 1);

  // Covered dims get a size that divides both the dim and the remaining
  // product.
  int64_t remaining = vector_length;
  for (int64_t i = rank - 1; i >= rank - covered_dims; --i) {
    const auto size = std::min(std::gcd(shape[i], remaining), shape[i]);
    tile_sizes[i] = size;
    remaining = std::max<int64_t>(remaining / size, 1);
  }

  return llvm::success();
}

/// Maps @p op to the default compute resource, unless it already is, and
/// annotates the vector width for @p element_type on it as the throttle.
///
/// @return The vector width.
auto mapToCompute(mlir::linalg::LinalgOp op, mlir::ktdf_arch::Mapping& mapping,
                  mlir::Type element_type) -> int64_t {
  auto compute =
      mapping.getOrMap(op, mapping.byKind().getDefaultCompute().getKind());
  assert(compute && "no default compute resource");

  LDBG() << "  compute: " << mlir::OpWithFlags(compute, kSkipRegions);

  // Determine the desired vector width on the compute resource.
  const auto simd_feature =
      compute.getFeature<mlir::ktdf_arch::feature::SIMD>();
  const auto vector_length =
      std::max(simd_feature.getLanes(element_type), int64_t{1});

  // Annotate the vector width as the throttle for this operation.
  setThrottle(op, vector_length);
  return vector_length;
}

auto determineTileSizes(mlir::linalg::LinalgOp op,
                        mlir::ktdf_arch::Mapping& mapping,
                        llvm::SmallVectorImpl<int64_t>& tile_sizes)
    -> llvm::LogicalResult {
  if (!op.hasPureTensorSemantics() || op->getNumResults() == 0) {
    return op->emitError("tiling error: expected pure tensor semantics");
  }
  const auto result_type =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!llvm::all_equal(op->getResultTypes())) {
    // Current implementation only allows for uniform tiling of all results.
    return op.emitError("tiling error: expected matching result types");
  }

  LDBG() << "determining tile sizes for "
         << mlir::OpWithFlags(op, kSkipRegions);

  // Map the op to a compute resource, if it isn't already.
  const auto vector_length =
      mapToCompute(op, mapping, result_type.getElementType());

  // The tile-sizes vector must cover every loop dimension (getNumLoops()).
  // Reduction dimensions are skipped (tile size = 0) so the tiling infra does
  // not create a loop for them.  Only parallel dimensions are tiled, using the
  // same rightmost-first shape-based algorithm as before.
  // Collect the loop bounds for each dimension by querying the op directly.
  // getStaticLoopRanges() scans all operand indexing maps so it works
  // regardless of whether any single operand has a full-rank map (e.g. when
  // an input has a broadcast/projection indexing map that skips some dims).
  const auto loop_ranges = op.getStaticLoopRanges();
  I64Vec parallel_loop_indices;
  I64Vec parallel_shape;
  for (auto [index, type] : llvm::enumerate(op.getIteratorTypesArray())) {
    if (type == mlir::utils::IteratorType::parallel) {
      parallel_loop_indices.push_back(index);
      parallel_shape.push_back(loop_ranges[index]);
    }
  }

  LDBG_OS([&](llvm::raw_ostream& os) {
    os << "  parallel_shape: {";
    llvm::interleaveComma(parallel_shape, os);
    os << "}";
  });

  // Compute tile sizes for the parallel dimensions.
  I64Vec parallel_tile_sizes;
  if (failed(::determineTileSizes(parallel_shape, vector_length,
                                  parallel_tile_sizes))) {
    return op->emitError("tiling error: inner dimensions not tileable");
  }

  // Assemble the final vector: 0 for reduction dims, computed size for
  // parallel dims.
  tile_sizes.resize(loop_ranges.size(), 0);
  for (int64_t k = 0; k < static_cast<int64_t>(parallel_loop_indices.size());
       ++k) {
    tile_sizes[parallel_loop_indices[k]] = parallel_tile_sizes[k];
  }

  LDBG_OS([&](llvm::raw_ostream& os) {
    os << "  tile_sizes: {";
    llvm::interleaveComma(tile_sizes, os);
    os << "}";
  });

  return llvm::success();
}

auto reifySize(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value,
               int64_t dim) -> mlir::FailureOr<mlir::OpFoldResult> {
  if (auto result = llvm::dyn_cast<mlir::OpResult>(value); result) {
    if (auto iface =
            result.getDefiningOp<mlir::ReifyRankedShapedTypeOpInterface>();
        iface) {
      return iface.reifyDimOfResult(builder, result.getResultNumber(), dim);
    }
  }

  const auto type = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!type) {
    return llvm::failure();
  }

  if (const auto size = type.getShape()[dim];
      !mlir::ShapedType::isDynamic(size)) {
    return mlir::OpFoldResult(builder.getI64IntegerAttr(size));
  }

  return mlir::OpFoldResult(
      mlir::tensor::DimOp::create(builder, loc, value, dim));
}

auto compressInput(mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp op,
                   mlir::OpOperand* input) -> llvm::LogicalResult {
  const auto type =
      llvm::dyn_cast<mlir::RankedTensorType>(input->get().getType());
  const auto bcast_dims = op.getMatchingIndexingMap(input).getBroadcastDims();
  if (!type || bcast_dims.empty()) {
    return llvm::failure();
  }

  const auto zero = rewriter.getI64IntegerAttr(0);
  const auto one = rewriter.getI64IntegerAttr(1);

  llvm::SmallVector<mlir::Range> ranges;
  ranges.reserve(type.getRank());
  for (auto dim : llvm::iota_range<int64_t>(0, type.getRank(), false)) {
    const auto size = reifySize(rewriter, op.getLoc(), input->get(), dim);
    if (failed(size)) {
      return llvm::failure();
    }
    ranges.push_back({zero, *size, one});
  }

  for (auto dim : bcast_dims) {
    ranges[dim] = {zero, one, zero};
  }

  input->set(mlir::tensor::ExtractSliceOp::create(rewriter, op.getLoc(),
                                                  input->get(), ranges));
  return llvm::success();
}

auto compressInputs(mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp op)
    -> llvm::LogicalResult {
  rewriter.startOpModification(op);
  auto changed = false;

  for (auto* const input : op.getDpsInputOperands()) {
    changed |= succeeded(compressInput(rewriter, op, input));
  }

  if (!changed) {
    rewriter.cancelOpModification(op);
    return llvm::failure();
  }

  LDBG() << "compressed " << mlir::OpWithFlags(op, kSkipRegions);
  rewriter.finalizeOpModification(op);
  return llvm::success();
}

auto tile(mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp& op,
          mlir::ktdf_arch::Mapping& mapping)
    -> llvm::FailureOr<mlir::scf::LoopVector> {
  rewriter.setInsertionPoint(op);

  // Remove needless dependencies on broadcasted dimensions.
  std::ignore = compressInputs(rewriter, op);

  // Determine tile sizes from output operand shape (needed for loop creation).
  I64Vec tile_sizes;
  if (failed(determineTileSizes(op, mapping, tile_sizes))) {
    return llvm::failure();
  }
  if (tile_sizes.empty()) {
    op.emitError("tiling error: no tileable dimensions");
    return llvm::failure();
  }

  // Tile the linalg.generic operation with the computed tile sizes
  mlir::linalg::LinalgTilingOptions tiling_options;
  tiling_options.setTileSizes(tile_sizes);
  tiling_options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);
  const auto maybe_tiling = tileLinalgOp(rewriter, op, tiling_options);
  if (failed(maybe_tiling)) {
    op->emitError("tiling error: unable to create loops");
    return llvm::failure();
  }

  // Annotate the created loops with their iterator type.
  {
    unsigned loop_index = 0;
    const auto iterator_types = op.getIteratorTypesArray();
    for (auto [dim, tile_size] : llvm::enumerate(tile_sizes)) {
      if (tile_size != 0) {
        // FIXME: Put the attribute name somewhere else.
        maybe_tiling->loops[loop_index++]->setAttr(
            "loop_type",
            getLoopTypeAttr(rewriter.getContext(), iterator_types[dim]));
      }
    }
  }

  // Replace the old with the tiled op.
  maybe_tiling->op->setDiscardableAttrs(op->getRawDictionaryAttrs());
  rewriter.replaceOp(op, maybe_tiling->tensorResults);
  op = maybe_tiling->op;

  return llvm::map_to_vector(maybe_tiling->loops, [](mlir::Operation* op) {
    return llvm::cast<mlir::scf::ForOp>(op);
  });
}

struct TileLoad : mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp> {
  explicit TileLoad(mlir::MLIRContext* context,
                    const AttrMapping& mem_space_map)
      : OpRewritePattern(context), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(mlir::tensor::ExtractSliceOp extract,
                       mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult override {
    auto source = llvm::dyn_cast<mlir::OpResult>(extract.getSource());
    if (!source) {
      return rewriter.notifyMatchFailure(extract, "no data tile source");
    }

    // Erase all the via hops on the way to the `ktdp.load`.
    llvm::SmallVector<mlir::Attribute> reverse_hops;
    auto load = llvm::dyn_cast<mlir::ktdp::LoadOp>(source.getOwner());
    while (!load) {
      if (auto via = llvm::dyn_cast<mlir::ktdf::ViaOp>(source.getOwner());
          via) {
        llvm::append_range(reverse_hops, llvm::reverse(via.getHops()));
        source = llvm::dyn_cast<mlir::OpResult>(via.getOperand());
        if (!source) {
          return rewriter.notifyMatchFailure(via, "no data tile source");
        }
        if (via->hasOneUse()) {
          rewriter.replaceOp(via, source);
        }
      } else {
        return rewriter.notifyMatchFailure(source.getOwner(),
                                           "not a data tile source");
      }

      load = llvm::dyn_cast<mlir::ktdp::LoadOp>(source.getOwner());
    }

    // Canonicalize the via hops, counting the load as a hop as well.
    reverse_hops.push_back(getMemorySpace(load.getAccessTile()));
    for (auto& hop : reverse_hops) {
      hop = mem_space_map_.map(hop);
    }
    reverse_hops.erase(std::unique(reverse_hops.begin(), reverse_hops.end()),
                       reverse_hops.end());

    // Create the `ktdp_lowering.load` that captures the applied tiling.
    auto new_load = mlir::ktdp_lowering::LoadOp::create(
        rewriter, load.getLoc(), extract.getType(), load.getAccessTile(),
        extract);
    setThrottle(new_load, getThrottle(extract->getOpResult(0)));

    // Add the hops back in after the load and replace `tensor.extract_slice`.
    mlir::Value loaded = new_load.getResult();
    if (auto reverse_via = llvm::MutableArrayRef(reverse_hops).drop_back(1);
        !reverse_via.empty()) {
      std::reverse(reverse_via.begin(), reverse_via.end());
      loaded = mlir::ktdf::ViaOp::create(rewriter, load.getLoc(), loaded,
                                         rewriter.getArrayAttr(reverse_via));
    }
    rewriter.replaceOp(extract, loaded);
    if (load->use_empty()) {
      rewriter.eraseOp(load);
    }
    return llvm::success();
  }

 private:
  const AttrMapping& mem_space_map_;
};

struct TileStore : mlir::OpRewritePattern<mlir::ktdp::StoreOp> {
  explicit TileStore(mlir::MLIRContext* context,
                     const AttrMapping& mem_space_map)
      : OpRewritePattern(context), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(mlir::ktdp::StoreOp store,
                       mlir::PatternRewriter& rewriter) const
      -> mlir::LogicalResult override {
    auto source = llvm::dyn_cast<mlir::OpResult>(store.getDataTile());
    if (!source) {
      return rewriter.notifyMatchFailure(store, "no data tile source");
    }

    // Erase all the via hops on the way to the `tensor.insert_slice`.
    mlir::scf::ForOp outermost;
    llvm::SmallVector<mlir::Attribute> reverse_hops{
        getMemorySpace(store.getAccessTile())};
    auto insert_slice =
        llvm::dyn_cast<mlir::tensor::InsertSliceOp>(source.getOwner());
    while (!insert_slice) {
      if (auto via = llvm::dyn_cast<mlir::ktdf::ViaOp>(source.getOwner());
          via) {
        llvm::append_range(reverse_hops, via.getHops());
        source = llvm::dyn_cast<mlir::OpResult>(via.getOperand());
        if (!source) {
          return rewriter.notifyMatchFailure(via, "no data tile source");
        }
        if (via->hasOneUse()) {
          rewriter.replaceOp(via, source);
        }
      } else if (auto loop =
                     llvm::dyn_cast<mlir::scf::ForOp>(source.getOwner());
                 loop) {
        outermost = outermost ? outermost : loop;
        source = llvm::dyn_cast<mlir::OpResult>(
            loop.getBody()->getTerminator()->getOperand(
                source.getResultNumber()));
        if (!source) {
          return rewriter.notifyMatchFailure(loop.getBody()->getTerminator(),
                                             "no data tile source");
        }
      } else {
        return rewriter.notifyMatchFailure(source.getOwner(),
                                           "not a data tile source");
      }

      insert_slice =
          llvm::dyn_cast<mlir::tensor::InsertSliceOp>(source.getOwner());
    }

    if (outermost) {
      if (llvm::any_of(outermost->getUsers(),
                       [&](mlir::Operation* user) -> bool {
                         return user->isBeforeInBlock(store);
                       })) {
        return rewriter.notifyMatchFailure(outermost,
                                           "loop has users before store");
      }
      rewriter.moveOpBefore(outermost, store);
    }

    // Canonicalize the via hops, counting the store as a hop as well.
    for (auto& hop : reverse_hops) {
      hop = mem_space_map_.map(hop);
    }
    reverse_hops.erase(std::unique(reverse_hops.begin(), reverse_hops.end()),
                       reverse_hops.end());

    // Determine the value to store. Conservatively, this must be the result of
    // insert_slice, since we didn't check how that is computed.
    mlir::Value stored = insert_slice.getResult();
    if (auto loop = llvm::dyn_cast<mlir::LoopLikeOpInterface>(
            insert_slice->getParentOp());
        loop) {
      auto dest_arg =
          llvm::dyn_cast<mlir::BlockArgument>(insert_slice.getDest());
      if (dest_arg && loop.getTiedLoopYieldedValue(dest_arg)->get() ==
                          insert_slice.getResult()) {
        // We know that insert_slice is a simple update on an iter_arg, which
        // means it can be erased.
        stored = insert_slice.getSource();
        rewriter.replaceAllUsesWith(insert_slice.getResult(),
                                    insert_slice.getDest());
      }
    }

    // Add the hops back in before the store.
    rewriter.setInsertionPointAfter(insert_slice);
    if (auto reverse_via = llvm::MutableArrayRef(reverse_hops).drop_front(1);
        !reverse_via.empty()) {
      std::reverse(reverse_via.begin(), reverse_via.end());
      stored = mlir::ktdf::ViaOp::create(rewriter, store.getLoc(), stored,
                                         rewriter.getArrayAttr(reverse_via));
    }

    // Create the `ktdp_lowering.store` that captures the applied tiling.
    auto new_store = mlir::ktdp_lowering::StoreOp::create(
        rewriter, store.getLoc(), stored, store.getAccessTile(), insert_slice);
    setThrottle(new_store, getThrottle(insert_slice.getSourceMutable()));
    rewriter.eraseOp(store);
    if (insert_slice->use_empty()) {
      rewriter.eraseOp(insert_slice);
    }
    return llvm::success();
  }

 private:
  const AttrMapping& mem_space_map_;
};

void breakLoopCarriedDependencies(mlir::RewriterBase& rewriter,
                                  mlir::DestinationStyleOpInterface op) {
  rewriter.setInsertionPoint(op);
  for (auto& outs : op.getDpsInitsMutable()) {
    auto slice = outs.get().getDefiningOp<mlir::tensor::ExtractSliceOp>();
    if (!slice || !llvm::isa<mlir::BlockArgument>(slice.getSource())) {
      continue;
    }

    // FIMXE: This only works because we aren't tiling reduction dimensions!

    rewriter.replaceOp(slice, mlir::tensor::EmptyOp::create(
                                  rewriter, op.getLoc(), slice.getMixedSizes(),
                                  slice.getType().getElementType()));
  }
}

void bubbleOpExtractSlice(mlir::RewriterBase& rewriter,
                          mlir::tensor::ExtractSliceOp op,
                          const AttrMapping& mem_space_map) {
  while (auto via = op.getSource().getDefiningOp<mlir::ktdf::ViaOp>()) {
    rewriter.modifyOpInPlace(op, [&]() { op.setOperand(0, via.getOperand()); });
    rewriter.modifyOpInPlace(via, [&]() {
      via.getResult().setType(op.getType());
      via.setOperand(op);
    });
    rewriter.replaceAllUsesExcept(op, via, via);
    rewriter.moveOpBefore(op, via);
  }

  if (auto load = op.getSource().getDefiningOp<mlir::ktdp_lowering::LoadOp>();
      load) {
    rewriter.modifyOpInPlace(op, [&]() {
      std::ignore = mlir::ktdf_arch::Mappable::setMapsTo(
          op, mlir::ktdf_arch::MapsToAttr::get(
                  rewriter.getContext(),
                  {llvm::cast<mlir::ktdf_arch::ResourceSpecAttr>(
                      mem_space_map.map(getMemorySpace(load.getSource())))}));
    });
  }
}

auto broadcastInput(mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp op,
                    mlir::OpOperand* input, const AttrMapping& mem_space_map)
    -> llvm::LogicalResult {
  const auto type =
      llvm::dyn_cast<mlir::RankedTensorType>(input->get().getType());
  const auto bcast_dims = op.getMatchingIndexingMap(input).getBroadcastDims();
  if (!type || bcast_dims.empty()) {
    return llvm::failure();
  }

  const auto vector_lanes =
      op->getAttrOfType<mlir::ktdf_arch::I64Attr>(kThrottleAttrName).getValue();
  const auto zero = rewriter.getI64IntegerAttr(0);
  const auto one = rewriter.getI64IntegerAttr(1);

  llvm::SmallVector<mlir::Range> ranges;
  ranges.reserve(type.getRank());
  int64_t elements = 1;
  for (auto dim : llvm::iota_range<int64_t>(0, type.getRank(), false)) {
    const auto size = reifySize(rewriter, op.getLoc(), input->get(), dim);
    if (failed(size)) {
      return llvm::failure();
    }
    ranges.push_back({zero, *size, one});

    if (llvm::is_contained(bcast_dims, dim)) {
      continue;
    }
    const auto static_size = mlir::getConstantIntValue(*size);
    if (!static_size ||
        llvm::MulOverflow(elements, *static_size, elements) != 0) {
      return llvm::failure();
    }
  }

  // The operand fills one vector, so only the lanes the other dimensions leave
  // are spread along the innermost broadcast dimension. A broadcast outside the
  // lanes keeps size 1.
  if (elements <= 0 || vector_lanes % elements != 0) {
    return llvm::failure();
  }
  const auto fill = vector_lanes / elements;
  if (fill == 1) {
    return llvm::failure();
  }

  for (auto dim : bcast_dims) {
    ranges[dim] = {zero, one, zero};
  }
  ranges[bcast_dims.back()] = {zero, rewriter.getI64IntegerAttr(fill), zero};

  auto broadcast = mlir::tensor::ExtractSliceOp::create(rewriter, op.getLoc(),
                                                        input->get(), ranges);
  input->set(broadcast);
  bubbleOpExtractSlice(rewriter, broadcast, mem_space_map);
  return llvm::success();
}

auto broadcastInputs(mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp op,
                     const AttrMapping& mem_space_map) -> llvm::LogicalResult {
  rewriter.startOpModification(op);
  auto changed = false;

  for (auto* const input : op.getDpsInputOperands()) {
    changed |= succeeded(broadcastInput(rewriter, op, input, mem_space_map));
  }

  if (!changed) {
    rewriter.cancelOpModification(op);
    return llvm::failure();
  }

  LDBG() << "broadcasted " << mlir::OpWithFlags(op, kSkipRegions);
  rewriter.finalizeOpModification(op);
  return llvm::success();
}

}  // namespace

void KTIRMapAndTilePass::runOnOperation() {
  if (disable_this_pass) {
    return;
  }

  mlir::func::FuncOp func = getOperation();
  mlir::IRRewriter rewriter(func);

  // Compute operations already nested in loops (e.g. by
  // IndirectAccessLoopMaterialization) are mapped but not tiled.
  auto computes = llvm::to_vector(func.getOps<mlir::linalg::LinalgOp>());
  llvm::SmallVector<mlir::linalg::LinalgOp> nested_computes;
  func.walk([&](mlir::linalg::LinalgOp op) {
    if (op->getParentOp() != func) {
      nested_computes.push_back(op);
    }
  });
  if (computes.empty() && nested_computes.empty()) {
    return;
  }

  // Obtain the default device and start a mapping.
  auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
  if (!default_device) {
    signalPassFailure();
    return;
  }
  mlir::ktdf_arch::Mapping mapping(default_device.getRef());

  // This pass supports remapping of the memory space attributes.
  AttrMapping mem_space_map;
  if (auto map = default_device->getAttrOfType<mlir::ktdf_arch::MapAttr>(
          "mem_space_mapping");
      map) {
    mem_space_map.insert_range(map);
  }

  for (auto op : nested_computes) {
    const auto result_type =
        op->getNumResults() == 1
            ? llvm::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType())
            : nullptr;
    if (!op.hasPureTensorSemantics() || !result_type) {
      op->emitError("mapping error: expected pure tensor semantics");
      signalPassFailure();
      return;
    }

    LDBG() << "mapping nested " << mlir::OpWithFlags(op, kSkipRegions);
    mapToCompute(op, mapping, result_type.getElementType());
  }

  // Tile all the compute operations and put them into loop nests that feature
  // loop-carried dependencies.
  llvm::SmallVector<mlir::scf::LoopVector> loop_nests;
  for (auto& op : computes) {
    auto maybe_loops = tile(rewriter, op, mapping);
    if (failed(maybe_loops)) {
      signalPassFailure();
      return;
    }
    if (maybe_loops->empty()) {
      LDBG() << "no loops generated for "
             << mlir::OpWithFlags(op, kSkipRegions);
      continue;
    }

    loop_nests.emplace_back(std::move(*maybe_loops));
    // Break the loop carried `tensor.extract_slice` dependencies.
    // FIXME: This is illegal if we're ever tiling across a reduction dimension.
    breakLoopCarriedDependencies(rewriter, op);
  }

  // Merge `tensor.(insert|extract)_slice` ops and tile `ktdp.(load|store)` and
  // remove all the unneeded `iter_args`.
  {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<TileLoad, TileStore>(patterns.getContext(), mem_space_map);
    mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    mlir::populateRegionBranchOpInterfaceCanonicalizationPatterns(
        patterns, mlir::scf::ForOp::getOperationName());

    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  // Broadcast compute operands back to target vector size.
  for (auto op : computes) {
    rewriter.setInsertionPoint(op);
    std::ignore = broadcastInputs(rewriter, op, mem_space_map);
  }

  // Clean up all the unused function-level constants that remain.
  for (auto& empty : llvm::make_early_inc_range(
           llvm::reverse(func.getFunctionBody().front()))) {
    if (mlir::isOpTriviallyDead(&empty)) {
      rewriter.eraseOp(&empty);
    }
  }
}
