//===------------------------------------------------------------*- c++ -*-===//
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
// Planner Resource Factories
//
// This file defines factory classes for managing private resources and
// transfer materialization information during path expansion planning.
//
// Key components:
// - PrivateResourceFactory: Creates and manages private resource specs
//   (memory buffers and FIFOs) and tracks their allocations
// - TransferInfoFactory: Creates and manages transfer materialization info
//   objects for data movement operations
//
// These factories use pointer-based references to avoid index-based schemes
// and provide a cleaner interface for the planner and materializer.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORM_PATHEXPANSION_PLANNERFACTORIES_H_
#define DATAFLOW_SCHEDULER_TRANSFORM_PATHEXPANSION_PLANNERFACTORIES_H_

#include <memory>

#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace scheduler {

// Forward declarations - actual definitions are in Planner.hpp
struct PrivateResourceSpec;
struct PrivateResourceAllocation;
struct TransferMaterializationInfo;

//===----------------------------------------------------------------------===//
// PrivateResourceFactory
//===----------------------------------------------------------------------===//

/// Factory for managing private resource allocations
/// This replaces index-based schemes with pointer-based references
class PrivateResourceFactory {
 public:
  /// Create a new memory buffer resource spec and return a pointer to it
  /// The factory owns the spec and keeps it alive
  /// @param memory_resource The memory resource where buffer will be allocated
  /// @param shape The shape of the memory buffer
  /// @param element_type The element type of the buffer
  /// @return Pointer to the created spec (owned by factory)
  PrivateResourceSpec* createMemoryBuffer(ResourceType memory_resource,
                                          llvm::ArrayRef<int64_t> shape,
                                          mlir::Type element_type);

  /// Create a new FIFO resource spec and return a pointer to it
  /// The factory owns the spec and keeps it alive
  /// @param fifo_type The FIFO type to allocate
  /// @param fifo_src Source attribute for the FIFO
  /// @param fifo_dest Destination attribute for the FIFO
  /// @param elements_per_slot Number of elements for each slot
  /// @param element_type The element type of the FIFO
  /// @return Pointer to the created spec (owned by factory)
  PrivateResourceSpec* createFifo(mlir::Attribute fifo_src,
                                  mlir::Attribute fifo_dest,
                                  llvm::ArrayRef<int64_t> elements_per_slot,
                                  mlir::Type element_type);

  /// Get all resource specs created by this factory
  /// @return Array reference to all owned specs
  llvm::ArrayRef<std::unique_ptr<PrivateResourceSpec>> getSpecs() const {
    return specs_;
  }

  /// Register an allocation for a spec (called during materialization)
  /// @param spec The spec to register allocation for
  /// @param ssa_values The SSA values representing the allocation
  void registerAllocation(const PrivateResourceSpec* spec,
                          llvm::ArrayRef<mlir::Value> ssa_values);

  /// Look up the allocation for a spec
  /// @param spec The spec to look up
  /// @return Pointer to allocation, or nullptr if not found
  const PrivateResourceAllocation* getAllocation(
      const PrivateResourceSpec* spec) const;

 private:
  // Owned resource specs
  llvm::SmallVector<std::unique_ptr<PrivateResourceSpec>> specs_;

  // Map from spec pointer to its allocation
  llvm::DenseMap<const PrivateResourceSpec*,
                 std::unique_ptr<PrivateResourceAllocation>>
      allocations_;
};

//===----------------------------------------------------------------------===//
// TransferInfoFactory
//===----------------------------------------------------------------------===//

/// Factory for creating and managing TransferMaterializationInfo objects
/// Owns all transfer info objects and provides factory methods
class TransferInfoFactory {
 public:
  /// Create transfer from template with intermediate buffer.
  /// Used when adapting an existing transfer to work with an intermediate
  /// memory resource. \p template_op must be a DataTransferOp or an
  /// IndDataTransferOp.
  /// @param template_op The original transfer operation to adapt
  /// @param edge The architecture edge this transfer implements
  /// @param intermediate_resource The intermediate memory resource
  /// @param current_resource The current stage's resource
  /// @param intermediate_is_source Whether intermediate is source or dest
  /// @param buffer_spec The buffer spec for the intermediate resource
  /// @param builder OpBuilder for creating attributes
  /// @return Pointer to created transfer info (owned by factory)
  TransferMaterializationInfo* createFromTemplateWithBuffer(
      mlir::Operation* template_op,
      const scheduler::arch_view::RoutingGraph::EdgeInfo& edge,
      ResourceType intermediate_resource, ResourceType current_resource,
      bool intermediate_is_source, const PrivateResourceSpec* buffer_spec,
      mlir::OpBuilder& builder);

  /// Create transfer from template that preserves both sides of the template.
  /// Indices, sizes and maps are copied verbatim and no private resources are
  /// attached; callers set source/dest private resources afterwards for the
  /// side(s) they want to retarget (e.g. a FIFO slot whose type changes).
  /// \p template_op must be a DataTransferOp.
  /// @param template_op The original transfer operation to adapt
  /// @param edge The architecture edge this transfer implements
  /// @param source_resource The source resource
  /// @param dest_resource The destination resource
  /// @return Pointer to created transfer info (owned by factory)
  TransferMaterializationInfo* createFromTemplate(
      mlir::Operation* template_op,
      const scheduler::arch_view::RoutingGraph::EdgeInfo& edge,
      ResourceType source_resource, ResourceType dest_resource);

  /// Create transfer for FIFO operation
  /// Used when adapting FIFO read/write operations to work with path expansion
  /// @param fifo_op The FIFO operation (read or write)
  /// @param edge The architecture edge this transfer implements
  /// @param source_resource The source resource
  /// @param dest_resource The destination resource
  /// @param fifo_spec The FIFO spec
  /// @param slot_index The slot index to use
  /// @param is_read Whether this is a read (true) or write (false)
  /// @return Pointer to created transfer info (owned by factory)
  TransferMaterializationInfo* createFromFifoOp(
      mlir::Operation* fifo_op,
      const scheduler::arch_view::RoutingGraph::EdgeInfo& edge,
      ResourceType source_resource, ResourceType dest_resource,
      const PrivateResourceSpec* fifo_spec, size_t slot_index, bool is_read);

  /// Create synthetic transfer for intermediate stage
  /// Used when creating new transfer operations for intermediate stages
  /// @param edge The architecture edge this transfer implements
  /// @param source_resource The source resource
  /// @param dest_resource The destination resource
  /// @param source_spec The source private resource spec
  /// @param source_slot_index The source slot index
  /// @param source_indices The source indices
  /// @param source_sizes The source sizes
  /// @param source_map The source affine map (from neighboring transfer)
  /// @param dest_spec The destination private resource spec
  /// @param dest_slot_index The destination slot index
  /// @param dest_indices The destination indices
  /// @param dest_sizes The destination sizes
  /// @param dest_map The destination affine map (from neighboring transfer)
  /// @param context MLIR context for creating attributes
  /// @return Pointer to created transfer info (owned by factory)
  TransferMaterializationInfo* createSynthetic(
      const scheduler::arch_view::RoutingGraph::EdgeInfo& edge,
      ResourceType source_resource, ResourceType dest_resource,
      const PrivateResourceSpec* source_spec, size_t source_slot_index,
      llvm::ArrayRef<mlir::Value> source_indices,
      llvm::ArrayRef<mlir::OpFoldResult> source_sizes,
      mlir::AffineMap source_map, const PrivateResourceSpec* dest_spec,
      size_t dest_slot_index, llvm::ArrayRef<mlir::Value> dest_indices,
      llvm::ArrayRef<mlir::OpFoldResult> dest_sizes, mlir::AffineMap dest_map,
      mlir::MLIRContext* context);

 private:
  // Owned transfer info objects
  llvm::SmallVector<std::unique_ptr<TransferMaterializationInfo>> transfers_;
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORM_PATHEXPANSION_PLANNERFACTORIES_H_s

// Made with Bob