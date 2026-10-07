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
// Planner Factory Implementations
// - PrivateResourceFactory: Manages private resource specs
// - TransferInfoFactory: Manages transfer materialization info
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Transforms/PathExpansion/PlannerFactories.h"

#include "dataflow-scheduler/Transforms/PathExpansion/Planner.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace scheduler;

namespace scheduler {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Helper to create an identity affine map for intermediate buffer accesses
/// This is used when we need an affine map for a memref buffer but don't have
/// one from a neighboring transfer (e.g., when the neighbor uses a FIFO)
static mlir::AffineMap createAffineMapForIntermediateBuffer(
    const PrivateResourceSpec* buffer, mlir::MLIRContext* context) {
  return mlir::AffineMap::getMultiDimIdentityMap(buffer->shape.size(), context);
}

/// Helper to create zero indices for an intermediate buffer
/// Used when a FIFO (which has no indices) is replaced by a memref buffer
static void createIndicesForIntermediateBuffer(
    const PrivateResourceSpec* buffer, mlir::Operation* contextOp,
    mlir::OpBuilder& builder, llvm::SmallVectorImpl<mlir::Value>& indices) {
  // Find the parent function to set insertion point
  auto funcOp = contextOp->getParentOfType<mlir::func::FuncOp>();
  assert(funcOp && "Expected operation to be within a function");

  // Set insertion point at the beginning of the function's entry block
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&funcOp.getBody().front());

  mlir::Value zero_index =
      mlir::arith::ConstantIndexOp::create(builder, contextOp->getLoc(), 0);
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    indices.push_back(zero_index);
  }
}

/// Helper to derive sizes from a private resource spec
/// Used to populate source_sizes or dest_sizes when they're empty
static void deriveSizesFromResourceSpec(
    const PrivateResourceSpec* spec, size_t slot_index,
    mlir::MLIRContext* context,
    llvm::SmallVectorImpl<mlir::OpFoldResult>& sizes) {
  if (!spec) return;

  mlir::OpBuilder builder(context);

  if (spec->kind == PrivateResourceSpec::Kind::kFifo) {
    // For FIFO, use the specific slot's capacity
    assert(slot_index < spec->elements_per_slot.size() &&
           "Slot index out of bounds");
    int64_t slot_capacity = spec->elements_per_slot[slot_index];
    sizes.push_back(builder.getI64IntegerAttr(slot_capacity));
  } else if (spec->kind == PrivateResourceSpec::Kind::kMemoryBuffer) {
    // For memory buffer, use the shape
    for (int64_t dim : spec->shape) {
      sizes.push_back(builder.getI64IntegerAttr(dim));
    }
  }
}

//===----------------------------------------------------------------------===//
// PrivateResourceFactory Implementation
//===----------------------------------------------------------------------===//

PrivateResourceSpec* PrivateResourceFactory::createMemoryBuffer(
    ResourceType memory_resource, llvm::ArrayRef<int64_t> shape,
    mlir::Type element_type) {
  auto spec = std::make_unique<PrivateResourceSpec>();
  spec->kind = PrivateResourceSpec::Kind::kMemoryBuffer;
  spec->memory_resource = memory_resource;
  spec->shape.assign(shape.begin(), shape.end());
  spec->element_type = element_type;

  PrivateResourceSpec* ptr = spec.get();
  specs_.push_back(std::move(spec));
  return ptr;
}

PrivateResourceSpec* PrivateResourceFactory::createFifo(
    mlir::Attribute fifo_src, mlir::Attribute fifo_dest,
    llvm::ArrayRef<int64_t> elements_per_slot, mlir::Type element_type) {
  auto spec = std::make_unique<PrivateResourceSpec>();
  spec->kind = PrivateResourceSpec::Kind::kFifo;
  spec->fifo_src = fifo_src;
  spec->fifo_dest = fifo_dest;
  spec->elements_per_slot.assign(elements_per_slot.begin(),
                                 elements_per_slot.end());
  spec->element_type = element_type;

  PrivateResourceSpec* ptr = spec.get();
  specs_.push_back(std::move(spec));
  return ptr;
}

void PrivateResourceFactory::registerAllocation(
    const PrivateResourceSpec* spec, llvm::ArrayRef<mlir::Value> ssa_values) {
  auto alloc = std::make_unique<PrivateResourceAllocation>(spec);
  alloc->ssa_values.assign(ssa_values.begin(), ssa_values.end());
  allocations_[spec] = std::move(alloc);
}

const PrivateResourceAllocation* PrivateResourceFactory::getAllocation(
    const PrivateResourceSpec* spec) const {
  auto it = allocations_.find(spec);
  return it != allocations_.end() ? it->second.get() : nullptr;
}

//===----------------------------------------------------------------------===//
// TransferInfoFactory Implementation
//===----------------------------------------------------------------------===//

namespace {
/// Per-operand indexing and layout metadata extracted from a template
/// DataTransferOp or IndDataTransferOp. Used when constructing multi-hop
/// transfers where one endpoint is an intermediate staging buffer and the other
/// endpoint preserves the original template's source or destination (FIFO or
/// memref).
struct TemplateTransferSides {
  // Indices/sizes/map for the source side of the template op.
  llvm::SmallVector<mlir::Value> src_indices;
  llvm::SmallVector<mlir::OpFoldResult> src_sizes;
  mlir::AffineMap src_map;
  // Indices/sizes/map for the dest side of the template op.
  llvm::SmallVector<mlir::Value> dst_indices;
  llvm::SmallVector<mlir::OpFoldResult> dst_sizes;
  mlir::AffineMap dst_map;
};

static TemplateTransferSides extractTemplateSides(mlir::Operation* op) {
  TemplateTransferSides s;
  if (auto dt = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op)) {
    s.src_indices.append(dt.getSourceIndices().begin(),
                         dt.getSourceIndices().end());
    s.src_sizes = dt.getMixedSourceSizes();
    s.src_map = dt.getSourceMapAttr() ? dt.getSourceMapAttr().getValue()
                                      : mlir::AffineMap();
    s.dst_indices.append(dt.getDestIndices().begin(),
                         dt.getDestIndices().end());
    s.dst_sizes = dt.getMixedDestSizes();
    s.dst_map = dt.getDestMapAttr() ? dt.getDestMapAttr().getValue()
                                    : mlir::AffineMap();
  } else {
    auto ind = mlir::cast<mlir::ktdf::IndDataTransferOp>(op);
    s.src_indices.append(ind.getDirSrcIndices().begin(),
                         ind.getDirSrcIndices().end());
    s.src_sizes = ind.getMixedDirSrcSizes();
    s.src_map = ind.getDirSrcMapAttr() ? ind.getDirSrcMapAttr().getValue()
                                       : mlir::AffineMap();
    s.dst_indices.append(ind.getDirDstIndices().begin(),
                         ind.getDirDstIndices().end());
    s.dst_sizes = ind.getMixedDirDstSizes();
    s.dst_map = ind.getDirDstMapAttr() ? ind.getDirDstMapAttr().getValue()
                                       : mlir::AffineMap();
  }
  return s;
}
}  // namespace

TransferMaterializationInfo* TransferInfoFactory::createFromTemplateWithBuffer(
    mlir::Operation* template_op,
    const scheduler::arch_view::RoutingGraph::EdgeInfo& edge,
    ResourceType intermediate_resource, ResourceType current_resource,
    bool intermediate_is_source, const PrivateResourceSpec* buffer_spec,
    mlir::OpBuilder& builder) {
  assert((mlir::isa<mlir::ktdf::DataTransferOp>(template_op) ||
          mlir::isa<mlir::ktdf::IndDataTransferOp>(template_op)) &&
         "template_op must be a DataTransferOp or IndDataTransferOp");

  auto sides = extractTemplateSides(template_op);

  auto transfer = std::make_unique<TransferMaterializationInfo>();
  transfer->template_op = template_op;
  transfer->hop = edge;
  transfer->source_resource =
      intermediate_is_source ? intermediate_resource : current_resource;
  transfer->dest_resource =
      intermediate_is_source ? current_resource : intermediate_resource;

  if (intermediate_is_source) {
    // Source is the intermediate buffer; dest side comes from the template.
    transfer->source_private_resource = buffer_spec;
    createIndicesForIntermediateBuffer(buffer_spec, template_op, builder,
                                       transfer->source_indices);
    deriveSizesFromResourceSpec(buffer_spec, 0, builder.getContext(),
                                transfer->source_sizes);
    transfer->source_map =
        createAffineMapForIntermediateBuffer(buffer_spec, builder.getContext());
    // Dest: copy indices, sizes, and map from template (may be memref or FIFO;
    // map is null for FIFO, affine map for memref).
    transfer->dest_indices = sides.dst_indices;
    transfer->dest_sizes = sides.dst_sizes;
    transfer->dest_map = sides.dst_map;
  } else {
    // Dest is the intermediate buffer; source side comes from the template.
    transfer->dest_private_resource = buffer_spec;
    // Source: copy indices, sizes, and map from template (may be memref or
    // FIFO; map is null for FIFO, affine map for memref).
    transfer->source_indices = sides.src_indices;
    transfer->source_sizes = sides.src_sizes;
    transfer->source_map = sides.src_map;
    createIndicesForIntermediateBuffer(buffer_spec, template_op, builder,
                                       transfer->dest_indices);
    deriveSizesFromResourceSpec(buffer_spec, 0, builder.getContext(),
                                transfer->dest_sizes);
    transfer->dest_map =
        createAffineMapForIntermediateBuffer(buffer_spec, builder.getContext());
  }

  TransferMaterializationInfo* result = transfer.get();
  transfers_.push_back(std::move(transfer));
  return result;
}

TransferMaterializationInfo* TransferInfoFactory::createFromTemplate(
    mlir::Operation* template_op,
    const scheduler::arch_view::RoutingGraph::EdgeInfo& edge,
    ResourceType source_resource, ResourceType dest_resource) {
  assert(mlir::isa<mlir::ktdf::DataTransferOp>(template_op) &&
         "template_op must be a DataTransferOp");

  auto sides = extractTemplateSides(template_op);

  auto transfer = std::make_unique<TransferMaterializationInfo>();
  transfer->template_op = template_op;
  transfer->hop = edge;
  transfer->source_resource = source_resource;
  transfer->dest_resource = dest_resource;
  transfer->source_indices = sides.src_indices;
  transfer->source_sizes = sides.src_sizes;
  transfer->source_map = sides.src_map;
  transfer->dest_indices = sides.dst_indices;
  transfer->dest_sizes = sides.dst_sizes;
  transfer->dest_map = sides.dst_map;

  TransferMaterializationInfo* result = transfer.get();
  transfers_.push_back(std::move(transfer));
  return result;
}

TransferMaterializationInfo* TransferInfoFactory::createFromFifoOp(
    mlir::Operation* fifo_op,
    const scheduler::arch_view::RoutingGraph::EdgeInfo& edge,
    ResourceType source_resource, ResourceType dest_resource,
    const PrivateResourceSpec* fifo_spec, size_t slot_index, bool is_read) {
  auto transfer = std::make_unique<TransferMaterializationInfo>();
  transfer->template_op = fifo_op;
  transfer->hop = edge;
  transfer->source_resource = source_resource;
  transfer->dest_resource = dest_resource;

  // Link the FIFO spec to the appropriate side
  if (is_read) {
    transfer->source_private_resource = fifo_spec;
    transfer->source_slot_index = slot_index;
    // Dest is memref, source is FIFO - no maps needed (will be set later when
    // we have indices)
  } else {
    transfer->dest_private_resource = fifo_spec;
    transfer->dest_slot_index = slot_index;
    // Source is memref, dest is FIFO - no maps needed (will be set later when
    // we have indices)
  }

  // Note: FIFO operations don't have indices and sizes - they work with tensors
  // The indices and sizes will remain empty for these operations
  // Affine maps will be set to null (no subscripts for FIFO side)

  TransferMaterializationInfo* result = transfer.get();
  transfers_.push_back(std::move(transfer));
  return result;
}

TransferMaterializationInfo* TransferInfoFactory::createSynthetic(
    const scheduler::arch_view::RoutingGraph::EdgeInfo& edge,
    ResourceType source_resource, ResourceType dest_resource,
    const PrivateResourceSpec* source_spec, size_t source_slot_index,
    llvm::ArrayRef<mlir::Value> source_indices,
    llvm::ArrayRef<mlir::OpFoldResult> source_sizes, mlir::AffineMap source_map,
    const PrivateResourceSpec* dest_spec, size_t dest_slot_index,
    llvm::ArrayRef<mlir::Value> dest_indices,
    llvm::ArrayRef<mlir::OpFoldResult> dest_sizes, mlir::AffineMap dest_map,
    mlir::MLIRContext* context) {
  auto transfer = std::make_unique<TransferMaterializationInfo>();
  transfer->template_op = nullptr;  // Synthetic transfer, no template
  transfer->hop = edge;
  transfer->source_resource = source_resource;
  transfer->dest_resource = dest_resource;
  transfer->source_private_resource = source_spec;
  transfer->source_slot_index = source_slot_index;
  transfer->source_indices.append(source_indices.begin(), source_indices.end());
  transfer->source_sizes.append(source_sizes.begin(), source_sizes.end());
  transfer->dest_private_resource = dest_spec;
  transfer->dest_slot_index = dest_slot_index;
  transfer->dest_indices.append(dest_indices.begin(), dest_indices.end());
  transfer->dest_sizes.append(dest_sizes.begin(), dest_sizes.end());

  // Derive sizes from resource specs if empty
  if (transfer->dest_sizes.empty() && dest_spec) {
    deriveSizesFromResourceSpec(dest_spec, dest_slot_index, context,
                                transfer->dest_sizes);
  }

  if (transfer->source_sizes.empty() && source_spec) {
    deriveSizesFromResourceSpec(source_spec, source_slot_index, context,
                                transfer->source_sizes);
  }

  // Use the affine maps passed from neighboring transfers
  transfer->source_map = source_map;
  transfer->dest_map = dest_map;

  TransferMaterializationInfo* result = transfer.get();
  transfers_.push_back(std::move(transfer));
  return result;
}

}  // namespace scheduler

// Made with Bob