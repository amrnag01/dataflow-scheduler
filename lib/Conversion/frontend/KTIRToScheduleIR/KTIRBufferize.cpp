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
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "Utils.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
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

struct KTIRBufferizePass
    : public impl::KTIRBufferizePassBase<KTIRBufferizePass> {
  using KTIRBufferizePassBase<KTIRBufferizePass>::KTIRBufferizePassBase;

  void runOnOperation() override;
};

struct LowerAccessTile
    : mlir::OpRewritePattern<mlir::ktdp::ConstructAccessTilesOp> {
  explicit LowerAccessTile(mlir::MLIRContext* context,
                           const AttrMapping& mem_space_map)
      : OpRewritePattern(context), mem_space_map_(mem_space_map) {}

  auto matchAndRewrite(mlir::ktdp::ConstructAccessTilesOp op,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    // Get the memory view (source memref) - first operand
    const auto memory_view = llvm::dyn_cast<MemRef>(op.getBase());
    if (!memory_view) {
      return rewriter.notifyMatchFailure(op, "not a memref");
    }
    if (auto* use = getSingleUse(op);
        !use || (llvm::isa<mlir::ktdp_lowering::LoadOp>(use->getOwner()) &&
                 llvm::isa<mlir::ktdp_lowering::StoreOp>(use->getOwner()))) {
      // FIXME: Use a DPS / bufferization interface.
      return rewriter.notifyMatchFailure(op, "can't bufferize users");
    }

    // Get the access tile indices and base_map. base_map projects the index
    // operands onto the per-dimension coordinates of the source memref, so
    // indices.size() == base_map.getNumInputs() and base_map.getNumResults()
    // == memref rank. For the common case base_map is identity.
    llvm::SmallVector<mlir::Value> raw_indices = op.getIndices();
    mlir::AffineMap base_map = op.getBaseMap();

    // Get the access tile result type to determine sizes
    const auto access_tile_type = op.getType();

    // Get tile dimensions from the access tile type
    llvm::ArrayRef<int64_t> tile_shape = access_tile_type.getShape();
    llvm::SmallVector<int64_t> tile_dims(tile_shape.begin(), tile_shape.end());

    // Get strides from memory view type
    llvm::SmallVector<int64_t> strides;
    if (auto strided_layout = mlir::dyn_cast<mlir::StridedLayoutAttr>(
            memory_view.getType().getLayout())) {
      strides.assign(strided_layout.getStrides().begin(),
                     strided_layout.getStrides().end());
    } else {
      // Default strides for row-major layout
      int64_t stride = 1;
      for (int i = memory_view.getType().getRank() - 1; i >= 0; --i) {
        strides.insert(strides.begin(), stride);
        stride *= memory_view.getType().getShape()[i];
      }
    }

    // Insert the cast sequence immediately before the (now-hoisted)
    // construct_access_tile.  Both the memory view and index operands have
    // already been moved above the outermost loop, so all operands dominate
    // this insertion point.

    // Apply base_map to materialize one index per source-memref dimension.
    // Operands to expandAffineMap are dim-values followed by symbol-values;
    // the op's symbol_operands feed both base_map symbols (if any) and the
    // access_tile_set, so they are appended after the raw indices.
    llvm::SmallVector<mlir::Value> expand_operands(raw_indices.begin(),
                                                   raw_indices.end());
    mlir::ValueRange symbol_operands = op.getSymbolOperands();
    expand_operands.append(symbol_operands.begin(), symbol_operands.end());
    std::optional<llvm::SmallVector<mlir::Value, 8>> per_dim_indices =
        mlir::affine::expandAffineMap(rewriter, op.getLoc(), base_map,
                                      expand_operands);
    if (!per_dim_indices) {
      return rewriter.notifyMatchFailure(op, "unable to expand base_map");
    }
    llvm::SmallVector<mlir::Value> indices(per_dim_indices->begin(),
                                           per_dim_indices->end());

    // After base_map expansion, indices.size() == memref rank == strides.size()
    if (indices.size() != strides.size()) {
      LDBG() << "Number of indices (" << indices.size()
             << ") does not match number of strides (" << strides.size() << ")";
      return rewriter.notifyMatchFailure(op, "invalid map");
    }

    // Calculate offset from per-dim indices and memory view strides
    mlir::Value offset =
        computeReinterpretCastOffset(rewriter, op.getLoc(), indices, strides);

    // Create sizes for reinterpret_cast
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    for (int64_t dim : tile_dims) {
      sizes.push_back(rewriter.getIndexAttr(dim));
    }

    // Create strides for reinterpret_cast (same as memory view)
    llvm::SmallVector<mlir::OpFoldResult> reinterpret_strides;
    for (int64_t stride : strides) {
      reinterpret_strides.push_back(rewriter.getIndexAttr(stride));
    }

    // Map the ktdp memory space to the device namespace using mem_space_mapping
    const auto memory_space = mem_space_map_.map(getMemorySpace(memory_view));
    // This propagates the mapped memory space to the reinterpret_cast
    const auto cast_source_type =
        mlir::MemRefType::get(memory_view.getType().getShape(),
                              memory_view.getType().getElementType(),
                              memory_view.getType().getLayout(), memory_space);
    auto memory_space_cast = mlir::memref::MemorySpaceCastOp::create(
        rewriter, op.getLoc(), cast_source_type, memory_view);

    llvm::SmallVector<int64_t> result_shape(tile_dims.begin(), tile_dims.end());
    const auto strided_layout = mlir::StridedLayoutAttr::get(
        rewriter.getContext(), mlir::ShapedType::kDynamic, strides);
    const auto result_type = mlir::MemRefType::get(
        result_shape, memory_view.getType().getElementType(), strided_layout,
        memory_space);

    mlir::OpFoldResult offset_fold_result(offset);
    auto cast_op = mlir::memref::ReinterpretCastOp::create(
        rewriter, op->getLoc(), result_type, memory_space_cast.getResult(),
        offset_fold_result, sizes, reinterpret_strides);
    // Replace access tile with reinterpret_cast
    rewriter.replaceOp(op, cast_op);
    return llvm::success();
  }

 private:
  AttrMapping mem_space_map_;

  [[nodiscard]] auto computeReinterpretCastOffset(
      mlir::OpBuilder& builder, mlir::Location loc,
      llvm::SmallVector<mlir::Value>& indices,
      llvm::SmallVector<int64_t>& strides) const -> mlir::Value {
    // Calculate offset from access tile indices and memory view strides.
    // For an access tile %A_view[%idx0, %idx1, ...] with strides [stride0,
    // stride1, ...], the offset is: %idx0 * stride0 + %idx1 * stride1 + ...
    // This is computed using a sequence of arith.muli and arith.addi
    // operations.

    size_t num_indices = indices.size();

    if (num_indices == 0) {
      // No indices means offset is 0
      return mlir::arith::ConstantIndexOp::create(builder, loc, 0);
    }

    // Compute terms: indices[i] * strides[i] for each dimension
    // Optimizations:
    // - Skip multiplication if stride is 1
    // - Skip addition if index is constant 0
    llvm::SmallVector<mlir::Value> terms;
    for (size_t i = 0; i < num_indices; ++i) {
      if (mlir::isZeroInteger(indices[i])) {
        continue;
      }

      mlir::Value term;
      if (strides[i] == 1) {
        term = indices[i];
      } else {
        mlir::Value stride_const =
            mlir::arith::ConstantIndexOp::create(builder, loc, strides[i]);
        term =
            mlir::arith::MulIOp::create(builder, loc, indices[i], stride_const);
      }
      terms.push_back(term);
    }

    if (terms.empty()) {
      // All indices were constant 0
      return mlir::arith::ConstantIndexOp::create(builder, loc, 0);
    }

    if (terms.size() == 1) {
      // Only one term, no addition needed
      return terms[0];
    }

    // Add all terms together
    mlir::Value offset = terms[0];
    for (size_t i = 1; i < terms.size(); ++i) {
      offset = mlir::arith::AddIOp::create(builder, loc, offset, terms[i]);
    }
    return offset;
  }
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
  patterns.add<LowerAccessTile>(patterns.getContext(), mem_space_map);

  if (failed(mlir::applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
    return;
  }
}
