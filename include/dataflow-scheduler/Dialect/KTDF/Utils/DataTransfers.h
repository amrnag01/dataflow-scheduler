//===-- DataTransfers.h -----------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_DATATRANSFERS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_DATATRANSFERS_H_

#include <llvm/ADT/DenseMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <cstdint>

#include "ktir/Dialect/KTDP/KTDPTypes.h"

namespace mlir {
class RewritePatternSet;
}  // namespace mlir

namespace scheduler {

/// Holds a value of a MemRefType.
using MemRef = mlir::TypedValue<mlir::MemRefType>;
/// Holds a value of an AccessTileType.
using AccessTile = mlir::TypedValue<mlir::ktdp::AccessTileType>;

/// Gets the memory space of @p memref , if any.
///
/// Follows to the ConstructMemoryViewOp source while KTDP doesn't preserve
/// memory spaces.
[[nodiscard]] auto getMemorySpace(MemRef memref) -> mlir::Attribute;
/// Gets the memory space of @p access_tile , if any.
[[nodiscard]] auto getMemorySpace(AccessTile access_tile) -> mlir::Attribute;
/// Gets the memory space of @p value , if any.
[[nodiscard]] auto getMemorySpace(mlir::Value value) -> mlir::Attribute;

/// Gets the throttle attribute value of @p op , or INT64_MAX if unset.
[[nodiscard]] auto getThrottle(mlir::Operation* op) -> int64_t;
/// Gets the maximum throttle with which @p result can be produced.
///
/// The maximum production throttle is the minimum of all user throttles.
[[nodiscard]] auto getThrottle(mlir::OpResult result) -> int64_t;
/// Gets the maximum throttle with which @p operand can be consumed.
///
/// The maximum consumption throttle is the producer's throttle, or INT64_MAX.
[[nodiscard]] auto getThrottle(mlir::OpOperand& operand) -> int64_t;

/// Sets the throttle attribute value of @p op .
void setThrottle(mlir::Operation* op, int64_t value);

/// Helper for storing a mapping between attributes.
struct AttrMapping : llvm::DenseMap<mlir::Attribute, mlir::Attribute> {
  using DenseMap::DenseMap;

  /// Maps @p attr to its image, or itself as the default.
  [[nodiscard]] auto map(mlir::Attribute attr) const -> mlir::Attribute {
    const auto mapped = lookup(attr);
    return mapped ? mapped : attr;
  }
};

/// Adds @p patterns that tile `ktdp.(load|store)` along the slices taken of
/// them, looking through via hops and reshapes that only add or drop unit
/// dimensions.
///
///  - Converts `tensor.extract_slice(ktdp.load)` to `ktdp_lowering.load`
///  - Converts `ktdp.store(tensor.insert_slice)` to `ktdp_lowering.store`
///
/// The memory spaces of the via hops are remapped using @p mem_space_map ,
/// which must outlive @p patterns .
void populateKTDPTilingPatterns(mlir::RewritePatternSet& patterns,
                                const AttrMapping& mem_space_map);

/// Adds @p patterns that coalesce data movement.
///
///  - Folds `tensor.(collapse|expand)_shape` into `ktdf.read_from_fifo`
///  - Folds `tensor.(collapse|expand)_shape` into `ktdf.write_to_fifo`
///  - Folds `ktdp_lowering.load` into its only `tensor.extract_slice` user.
void populateDataMovementCoalescingPatterns(mlir::RewritePatternSet& patterns);

/// Adds @p patterns that convert various ops to `ktdf.data_transfer`.
///
///  - Converts `ktdf.write_to_fifo(ktdp_lowering.load)`
///  - Converts `ktdp_lowering.store(ktdf.read_from_fifo)`
///  - Converts `ktdp_lowering.store(ktdp_lowering.load)`
///  - Converts `ktdf.write_to_fifo(ktdp_lowering.ind_load)` to
///    `ktdf.ind_data_transfer`
///  - Converts `ktdp_lowering.ind_store(ktdf.read_from_fifo)` to
///    `ktdf.ind_data_transfer`
///
/// This includes the data movement coalescing patterns.
void populateConvertToDataTransferPatterns(mlir::RewritePatternSet& patterns);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_UTILS_DATATRANSFERS_H_
