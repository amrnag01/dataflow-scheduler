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
// AddressAssignment: Analyze memory allocations and compute sizes for
// address assignment.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Analysis/Presburger/IntegerRelation.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>  // IWYU pragma: keep
#include <mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/AlignmentAttrInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ValueBoundsOpInterface.h>
#include <mlir/Pass/Pass.h>

#include <memory>
#include <optional>

#include "dataflow-scheduler/Analysis/MemoryTrackerAnalysis.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFDialect.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"

#define PASS_NAME "address-assignment"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Address Assignment pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_ADDRESSASSIGNMENTPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Tries to determine the upper bound of @p shape given @p dynamic_sizes .
[[nodiscard]] auto getUpperBound(llvm::ArrayRef<int64_t> shape,
                                 mlir::ValueRange dynamic_sizes)
    -> llvm::FailureOr<llvm::SmallVector<int64_t>> {
  llvm::SmallVector<int64_t> result;

  unsigned dynamic_index = 0U;
  for (auto [idx, dim] : llvm::enumerate(shape)) {
    if (mlir::ShapedType::isStatic(dim)) {
      result.push_back(dim);
      continue;
    }

    const auto size = dynamic_sizes[dynamic_index++];
    LDBG(1) << "  Dynamic dim " << idx << ": SSA value " << size;

    const auto maybe_bound =
        mlir::ValueBoundsConstraintSet::computeConstantBound(
            mlir::presburger::BoundType::UB, size, nullptr, true);
    if (llvm::failed(maybe_bound)) {
      LDBG(1) << "    Failed to compute constant upper bound";
      LDBG(1)
          << "    Note: ValueBoundsConstraintSet could not resolve the bound "
             "to a constant value.";
      LDBG(1) << "    This may require more sophisticated analysis or manual "
                 "annotation.";
      return llvm::failure();
    }

    LDBG(1) << "    Computed upper bound: " << *maybe_bound;
    result.push_back(*maybe_bound);
  }

  return std::move(result);
}

/// Named constraint for an allocation that is processed by this pass.
struct AllocOp
    : mlir::Op<AllocOp, mlir::OpTrait::ZeroRegions, mlir::OpTrait::OneResult,
               mlir::OpTrait::OneTypedResult<mlir::MemRefType>::Impl,
               mlir::MemoryEffectOpInterface::Trait> {
  [[nodiscard]] static auto classof(mlir::Operation* op) -> bool {
    return llvm::isa<mlir::memref::AllocOp, mlir::memref::AllocaOp>(op);
  }

  using Op::Op;

  /*implicit*/ AllocOp(mlir::memref::AllocOp alloc) : Op(alloc) {}
  /*implicit*/ AllocOp(mlir::memref::AllocaOp alloc) : Op(alloc) {}

  void getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                      mlir::MemoryEffects::Effect>>& effects) {
    llvm::cast<mlir::MemoryEffectOpInterface>(getOperation())
        .getEffects(effects);
  }

  /// Tries to get the dynamic sizes associated with this allocation, if any.
  [[nodiscard]] auto getDynamicSizes() -> llvm::FailureOr<mlir::OperandRange> {
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(getOperation())) {
      return alloc.getDynamicSizes();
    }
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocaOp>(getOperation())) {
      return alloc.getDynamicSizes();
    }
    return llvm::failure();
  }

  /// Tries to get the static upper bound for the allocated memref shape.
  [[nodiscard]] auto getUpperShapeBound()
      -> llvm::FailureOr<llvm::SmallVector<int64_t>> {
    if (getType().hasStaticShape()) {
      return llvm::to_vector(getType().getShape());
    }

    const auto maybe_sizes = getDynamicSizes();
    if (llvm::failed(maybe_sizes)) {
      return llvm::failure();
    }
    return getUpperBound(getType().getShape(), *maybe_sizes);
  }

  /// Tries to get the static upper bound for the largest element offset.
  [[nodiscard]] auto getUpperOffsetBound() -> llvm::FailureOr<size_t> {
    auto maybe_shape = getUpperShapeBound();
    if (llvm::failed(maybe_shape)) {
      return llvm::failure();
    }

    llvm::SmallVector<int64_t> strides;
    int64_t offset;
    if (llvm::failed(getType().getLayout().getStridesAndOffset(
            *maybe_shape, strides, offset)) ||
        offset < 0 ||
        llvm::any_of(strides, [](int64_t stride) { return stride < 0; })) {
      LDBG() << "unsupported layout: " << getType().getLayout();
      return llvm::failure();
    }

    auto result = static_cast<size_t>(offset);
    for (auto [sz, stride] : llvm::zip_equal(*maybe_shape, strides)) {
      size_t temp;
      if (__builtin_mul_overflow(sz - 1, stride, &temp) ||
          __builtin_add_overflow(result, temp, &result)) {
        LDBG() << "allocation too large";
        return llvm::failure();
      }
    }
    return result;
  }

  /// Tries to get the static lower bound for the allocation size.
  [[nodiscard]] auto getSize() -> llvm::FailureOr<size_t> {
    LDBG(1) << "Analyzing allocation: " << *this;
    LDBG(1) << "  Type: " << getResult().getType();

    const auto maybe_offset = getUpperOffsetBound();
    if (llvm::failed(maybe_offset)) {
      return llvm::failure();
    }
    const auto maybe_size =
        tryGetSizeInBytes(getResult().getType().getElementType());
    if (!maybe_size) {
      LDBG() << "unsupported element type";
      return llvm::failure();
    }

    size_t total_bytes;
    if (__builtin_mul_overflow(*maybe_offset + 1, *maybe_size, &total_bytes)) {
      LDBG() << "allocation too large";
      return llvm::failure();
    }

    LDBG(1) << "  Total size: " << total_bytes << " bytes (" << *maybe_offset
            << " elements * " << *maybe_size << " bytes/element)";
    return total_bytes;
  }

  /// Gets the alignment requirement, if any.
  [[nodiscard]] auto getAlignment() -> llvm::MaybeAlign {
    if (auto iface =
            llvm::dyn_cast<mlir::AlignmentAttrOpInterface>(getOperation());
        iface) {
      return iface.getMaybeAlign();
    }
    return {};
  }
};

/// Collect all memref.alloc operations that have a memory space attribute.
[[nodiscard]] auto collectAllocations(mlir::ModuleOp module)
    -> llvm::SmallVector<AllocOp> {
  llvm::SmallVector<AllocOp> allocs;

  module.walk([&](AllocOp alloc) {
    auto memref_type = alloc.getType();
    if (memref_type.getMemorySpace()) {
      allocs.push_back(alloc);
    } else {
      LDBG(1) << "Skipping allocation with missing memory space: " << alloc;
    }
    return mlir::WalkResult::advance();
  });

  return allocs;
}

/// Result of processing a single allocation.
struct AllocationResult {
  bool success;
  size_t assigned_address;
  size_t size;
};

/// Replace memref.alloc with unrealized_conversion_cast using the assigned
/// address, and remove any corresponding memref.dealloc operations.
void materializeAddressAssignment(AllocOp alloc, size_t assigned_address,
                                  mlir::OpBuilder& builder) {
  builder.setInsertionPoint(alloc);

  auto alloc_type = alloc.getType();

  // Create constant index for the assigned address
  mlir::Value addr_value = mlir::arith::ConstantIndexOp::create(
      builder, alloc.getLoc(), assigned_address);

  // Create unrealized_conversion_cast from index to memref
  auto cast = mlir::UnrealizedConversionCastOp::create(builder, alloc.getLoc(),
                                                       alloc_type, addr_value);

  LDBG(1) << "  Materialized address assignment:";
  LDBG(1) << "    Original alloc: " << alloc;
  LDBG(1) << "    Replaced with: " << cast;

  // Collect all dealloc operations that use this allocation
  llvm::SmallVector<mlir::memref::DeallocOp> deallocs_to_remove;
  for (mlir::Operation* user : alloc->getUsers()) {
    if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(user)) {
      deallocs_to_remove.push_back(dealloc);
    }
  }

  // Replace all uses of the alloc with the cast
  alloc.replaceAllUsesWith(cast.getResult(0));

  // Remove the dealloc operations
  for (auto dealloc : deallocs_to_remove) {
    LDBG(1) << "    Removing dealloc: " << dealloc;
    dealloc.erase();
  }

  // Erase the original alloc
  alloc.erase();
}

struct AddressAssignmentPass
    : public impl::AddressAssignmentPassBase<AddressAssignmentPass> {
  AddressAssignmentPass()
      : scheduler_ctx_(SchedulerExtContext::dummyContext()) {}
  explicit AddressAssignmentPass(const SchedulerExtContext& ctx)
      : scheduler_ctx_(ctx) {}

  void runOnOperation() override {
    if (DisableThisPass) {
      return;
    }

    LDBG(1) << "========= " PASS_NAME " =========";

    mlir::ModuleOp module = getOperation();

    LDBG(1) << "Starting address assignment analysis";

    // Collect all allocations with memory space attributes
    auto allocs = collectAllocations(module);
    LDBG(1) << "Found " << allocs.size()
            << " allocations with memory space attributes";
    if (allocs.empty()) {
      return;
    }

    // Get the MemoryTrackerAnalysis.
    auto& tracker = getAnalysis<MemoryTrackerAnalysis>();
    mlir::OpBuilder builder(&getContext());

    LDBG(1) << "Processing allocations with MemoryTracker:";

    for (auto alloc : allocs) {
      if (failed(processAllocation(alloc, tracker, builder))) {
        ++num_failed;
      }
      ++num_processed;
    }

    LDBG(1) << "Allocation summary:";
    LDBG(1) << "  Total allocations: " << num_processed;
    LDBG(1) << "  Failed assignments: " << num_failed;
    // TODO: Use memory hierarchy view to diagnose memref.alloc in namespaces
    // that are not supposed to be address assigned by the scheduler

    if (num_failed > 0) {
      module->emitError("AddressAssignment: failed to assign addresses for ")
          << num_failed << " allocation(s)";
      signalPassFailure();
    }
  }

 private:
  auto processAllocation(AllocOp alloc, MemoryTrackerAnalysis& tracker,
                         mlir::OpBuilder& builder)
      -> llvm::FailureOr<AllocationResult> {
    const auto maybe_size = alloc.getSize();
    if (llvm::failed(maybe_size)) {
      LDBG(1) << "  Skipping allocation with unknown size: " << alloc;
      return llvm::failure();
    }

    // Get the memory space attribute from the memref type
    auto memref_type = alloc.getType();
    mlir::Attribute memory_space_attr = memref_type.getMemorySpace();
    assert(memory_space_attr);

    const auto alignment = alloc.getAlignment().valueOrOne();
    auto address_result =
        tracker.allocate(memory_space_attr, *maybe_size, alignment.value());

    if (!address_result) {
      llvm::Error err = address_result.takeError();
      std::string error_msg;
      llvm::handleAllErrors(std::move(err), [&](const llvm::ErrorInfoBase& ei) {
        error_msg = ei.message();
      });

      alloc.emitError() << "Failed to allocate " << memory_space_attr
                        << " memory: " << error_msg;
      return llvm::failure();
    }

    size_t assigned_address = *address_result;

    LDBG(1) << "  Assigned address " << assigned_address
            << " to allocation: " << alloc;
    LDBG(1) << "    Size: " << *maybe_size
            << " bytes, Alignment: " << alignment.value() << " bytes";

    // Materialize the address assignment in the IR
    materializeAddressAssignment(alloc, assigned_address, builder);

    return AllocationResult{true, assigned_address, *maybe_size};
  }

  const SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createAddressAssignmentPass() {
  return std::make_unique<AddressAssignmentPass>();
}

std::unique_ptr<mlir::Pass> scheduler::createAddressAssignmentPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<AddressAssignmentPass>(scheduler_ctx);
}

// Made with Bob
