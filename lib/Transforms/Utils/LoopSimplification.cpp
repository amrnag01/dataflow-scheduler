//===-- LoopSimplification.cpp -----------------------------------*- c++ -*-==//
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

#include "dataflow-scheduler/Transforms/Utils/LoopSimplification.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Transforms.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

using namespace scheduler;

//===----------------------------------------------------------------------===//
// LoopNestConstraints
//===----------------------------------------------------------------------===//

auto LoopNestConstraints::populateIndices(
    mlir::OffsetSizeAndStrideOpInterface slice)
    -> mlir::FailureOr<llvm::SmallVector<unsigned>> {
  using mlir::presburger::BoundType;

  if (const auto it = slice_points_.find(slice.getOperation());
      it != slice_points_.end()) {
    return it->second;
  }
  const auto strides = mlir::getConstantIntValues(slice.getMixedStrides());
  if (!strides) {
    return llvm::failure();
  }

  // index = offset + stride * witness, 0 <= witness < size
  llvm::SmallVector<unsigned> point;
  for (auto [offset, size, stride] : llvm::zip_equal(
           slice.getMixedOffsets(), slice.getMixedSizes(), *strides)) {
    const auto witness = insert();
    addBound(BoundType::LB, witness, getExpr(0));
    addBound(BoundType::UB, witness, getExpr(size));
    point.push_back(insert());
    addBound(BoundType::EQ, point.back(),
             getExpr(offset) + stride * getPosExpr(witness));
  }
  processWorklist();
  addLoopLattices();
  return slice_points_[slice.getOperation()] = std::move(point);
}

auto LoopNestConstraints::pair(mlir::LoopLikeOpInterface loop,
                               llvm::SmallVectorImpl<unsigned>& second) const
    -> mlir::presburger::IntegerPolyhedron {
  using mlir::presburger::VarKind;

  mlir::presburger::IntegerPolyhedron first(cstr);
  auto other = first;
  for (unsigned column = 0; column < cstr.getNumDimAndSymbolVars(); ++column) {
    const auto& value_dim = positionToValueDim[column];
    if (value_dim && loop.isDefinedOutsideOfLoop(value_dim->first)) {
      second.push_back(column);
      continue;
    }
    second.push_back(first.appendVar(VarKind::Symbol));
    other.appendVar(VarKind::Symbol);
    other.swapVar(column, second.back());
  }
  first.appendVar(VarKind::Local, cstr.getNumLocalVars());
  other.insertVar(VarKind::Local, 0, cstr.getNumLocalVars());
  first.append(other);
  return first;
}

void LoopNestConstraints::addLoopLattices() {
  // Populating the lower bound may add more IVs at the end.
  for (std::size_t column = 0; column < positionToValueDim.size(); ++column) {
    const auto value_dim = positionToValueDim[column];
    if (!value_dim || value_dim->second != kIndexValue) {
      continue;
    }
    auto loop = mlir::scf::getForInductionVarOwner(value_dim->first);
    if (!loop || !lattice_loops_.insert(loop).second) {
      continue;
    }
    const auto step = mlir::getConstantIntValue(loop.getStep());
    if (!step) {
      continue;
    }

    const auto lower_bound = getExpr(loop.getLowerBound());
    const auto iteration = getPosExpr(insert());
    bound(loop.getInductionVar()) == lower_bound + *step* iteration;
    processWorklist();
  }
}

//===----------------------------------------------------------------------===//
// scheduler::populateSimplifyLoopsPatterns
//===----------------------------------------------------------------------===//

namespace {

struct DropUnusedInsertSlice
    : mlir::OpRewritePattern<mlir::tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::tensor::InsertSliceOp insert_slice,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto arg = llvm::dyn_cast<mlir::BlockArgument>(insert_slice.getDest());
    if (!arg) {
      return rewriter.notifyMatchFailure(insert_slice,
                                         "not inserting into iter_arg");
    }
    auto loop = llvm::dyn_cast<mlir::scf::ForOp>(arg.getOwner()->getParentOp());
    if (!loop) {
      return rewriter.notifyMatchFailure(insert_slice,
                                         "not inserting into iter_arg");
    }

    // Result must be yielded to the next iteration, possibly via more inserts.
    const auto* result = loop.getTiedLoopYieldedValue(arg);
    while (result->get() != insert_slice) {
      auto insert = result->get().getDefiningOp<mlir::tensor::InsertSliceOp>();
      if (!insert) {
        return rewriter.notifyMatchFailure(result->getOwner(),
                                           "not inserting into iter_arg");
      }
      result = &insert.getDestMutable();
    }

    // Collect all the loops that surround this insert_slice, and follow the
    // final value out of the loop nest.
    LoopNestConstraints constraints(getContext());
    llvm::SmallVector<unsigned> iv_columns;
    iv_columns.push_back(constraints.populate(loop.getInductionVar()));
    auto outermost = loop;
    auto final_value = loop.getTiedLoopResult(arg);
    for (auto parent = loop->getParentOfType<mlir::scf::ForOp>(); parent;
         parent = parent->getParentOfType<mlir::scf::ForOp>()) {
      auto siblings = parent.getBody()->without_terminator();
      if (std::next(siblings.begin()) != siblings.end()) {
        return rewriter.notifyMatchFailure(parent, "imperfect loop nest");
      }
      if (!final_value.hasOneUse() || final_value.use_begin()->getOwner() !=
                                          parent.getBody()->getTerminator()) {
        return rewriter.notifyMatchFailure(insert_slice,
                                           "result escapes the loop nest");
      }
      final_value =
          parent->getResult(final_value.use_begin()->getOperandNumber());
      iv_columns.push_back(constraints.populate(parent.getInductionVar()));
      outermost = parent;
    }
    std::reverse(iv_columns.begin(), iv_columns.end());

    // The final value must not be observed after the loop nest.
    if (!final_value.use_empty()) {
      return rewriter.notifyMatchFailure(insert_slice,
                                         "result escapes the loop nest");
    }

    // Get the vector of dimensions that corresponds to the writing iteration.
    const auto insert_point = constraints.populateIndices(insert_slice);
    if (failed(insert_point)) {
      return rewriter.notifyMatchFailure(insert_slice,
                                         "unable to determine bounds");
    }

    const auto is_independent =
        [&](mlir::tensor::ExtractSliceOp extract_slice) -> bool {
      const auto add_equal = [](mlir::presburger::IntegerRelation& relation,
                                unsigned lhs, unsigned rhs) {
        llvm::SmallVector<int64_t> row(relation.getNumCols(), 0);
        row[lhs] += 1;
        row[rhs] -= 1;
        relation.addEquality(row);
      };

      // Get the vector of dimensions that corrsponds to the reading iteration.
      const auto extract_point = constraints.populateIndices(extract_slice);
      if (failed(extract_point)) {
        return false;
      }

      // Equate the tensor coordinates and determine whether they only overlap
      // in the same iteration.
      llvm::SmallVector<unsigned> second;
      auto overlap = constraints.pair(outermost, second);
      auto same_iteration = mlir::presburger::IntegerPolyhedron::getUniverse(
          overlap.getSpace().getSpaceWithoutLocals());
      for (auto [lhs, rhs] : llvm::zip_equal(*extract_point, *insert_point)) {
        add_equal(overlap, lhs, second[rhs]);
      }
      for (auto column : iv_columns) {
        add_equal(same_iteration, column, second[column]);
      }
      return overlap.isSubsetOf(same_iteration);
    };

    // Check if none of the users observes the result of the insertion.
    for (auto* const user : arg.getUsers()) {
      if (llvm::isa<mlir::tensor::InsertSliceOp>(user)) {
        continue;
      }
      auto extract_slice = llvm::dyn_cast<mlir::tensor::ExtractSliceOp>(user);
      if (!extract_slice) {
        return rewriter.notifyMatchFailure(user,
                                           "use is not `tensor.extract_slice`");
      }
      if (!is_independent(extract_slice)) {
        return rewriter.notifyMatchFailure(extract_slice,
                                           "user is not independent");
      }
    }

    // Simply drop the insertion.
    rewriter.replaceOp(insert_slice, insert_slice.getDest());
    return llvm::success();
  }
};

}  // namespace

void scheduler::populateSimplifyLoopsPatterns(
    mlir::RewritePatternSet& patterns) {
  mlir::populateRegionBranchOpInterfaceCanonicalizationPatterns(
      patterns, mlir::scf::ForOp::getOperationName());

  patterns.add<DropUnusedInsertSlice>(patterns.getContext());
}
