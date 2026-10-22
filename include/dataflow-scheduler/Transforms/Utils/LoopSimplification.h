//===-- LoopSimplification.h -------------------------------------*- c++ -*-==//
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

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_LOOPSIMPLIFICATION_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_LOOPSIMPLIFICATION_H_

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/Analysis/Presburger/IntegerRelation.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/ValueBoundsOpInterface.h>

namespace mlir {

class RewritePatternSet;

}  // namespace mlir

namespace scheduler {

/// Constraints on index values, as populated through `ValueBoundsOpInterface`,
/// strengthened by the iteration lattice `lb + k * step` of `scf.for` loops.
/// Can be shared between queries as long as the IR isn't modified.
class LoopNestConstraints : protected mlir::ValueBoundsConstraintSet {
 public:
  explicit LoopNestConstraints(mlir::MLIRContext* context,
                               const StopConditionFn& stop_condition)
      : ValueBoundsConstraintSet(context, stop_condition) {}
  explicit LoopNestConstraints(mlir::MLIRContext* context)
      : LoopNestConstraints(
            context,
            [](mlir::Value, std::optional<int64_t>,
               ValueBoundsConstraintSet&) -> bool { return false; }) {}

  /// Adds constraints for the index-typed @p value and its backward slice.
  /// @return the column of @p value.
  auto populate(mlir::Value value) -> unsigned {
    populateConstraints(value, std::nullopt);
    addLoopLattices();
    return getPos(value);
  }

  /// Adds columns that range over the indices of @p slice, which must have
  /// static strides.
  /// @return the column of each dimension of the index.
  auto populateIndices(mlir::OffsetSizeAndStrideOpInterface slice)
      -> mlir::FailureOr<llvm::SmallVector<unsigned>>;

  /// Instantiates the constraints twice, sharing the columns of values defined
  /// outside of @p loop. The first instance keeps the columns of the
  /// constraints, @p second maps them to those of the second instance.
  [[nodiscard]] auto pair(mlir::LoopLikeOpInterface loop,
                          llvm::SmallVectorImpl<unsigned>& second) const
      -> mlir::presburger::IntegerPolyhedron;

 private:
  /// Constrains every populated `scf.for` IV with a constant step to its
  /// iteration lattice, which `ValueBoundsOpInterface` doesn't express.
  void addLoopLattices();

  llvm::DenseSet<mlir::Operation*> lattice_loops_;
  llvm::DenseMap<mlir::Operation*, llvm::SmallVector<unsigned>> slice_points_;
};

void populateSimplifyLoopsPatterns(mlir::RewritePatternSet& patterns);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_LOOPSIMPLIFICATION_H_
