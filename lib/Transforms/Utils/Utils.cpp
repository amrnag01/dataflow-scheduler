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
// Utility functions for transform passes.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Transforms/Utils/Utils.h"

#include <cassert>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace scheduler;

bool scheduler::isTargetConstant(int target, mlir::Value val) {
  if (auto const_op = val.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    return const_op.value() == target;
  }
  return false;
}

std::optional<int64_t> scheduler::getConstantIndexValue(mlir::Value value) {
  if (auto const_op = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    return const_op.value();
  }
  return std::nullopt;
}

std::optional<int64_t> scheduler::getStaticTripCount(mlir::scf::ForOp loop) {
  if (auto tc = loop.getStaticTripCount()) return tc->getSExtValue();
  return std::nullopt;
}

mlir::LogicalResult scheduler::cleanupPrivateOp(
    mlir::ktdf::PrivateOp private_op) {
  mlir::Block* body = private_op.getBody();
  auto yield_op = private_op.getYieldOp();

  llvm::SmallVector<unsigned> live_result_indices;
  llvm::SmallVector<mlir::Type> live_result_types;
  llvm::SmallVector<mlir::Value> live_yield_operands;

  for (auto [idx, result] : llvm::enumerate(private_op.getResults())) {
    if (result.use_empty()) continue;
    live_result_indices.push_back(idx);
    live_result_types.push_back(result.getType());
    live_yield_operands.push_back(yield_op.getOperands()[idx]);
  }

  if (live_result_indices.size() == private_op.getNumResults()) {
    return mlir::success();
  }

  llvm::SetVector<mlir::Operation*> needed_ops;
  llvm::SmallVector<mlir::Value> worklist(live_yield_operands.begin(),
                                          live_yield_operands.end());

  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    auto op_result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!op_result) continue;

    mlir::Operation* def_op = op_result.getOwner();
    if (def_op->getBlock() != body || def_op == yield_op) continue;
    if (!needed_ops.insert(def_op)) continue;

    for (mlir::Value operand : def_op->getOperands()) {
      worklist.push_back(operand);
    }
  }

  mlir::OpBuilder builder(private_op);
  auto new_private = mlir::ktdf::PrivateOp::create(builder, private_op.getLoc(),
                                                   live_result_types);
  mlir::Block& new_body = *new_private.getBody();

  mlir::IRMapping mapping;
  builder.setInsertionPointToStart(&new_body);
  for (mlir::Operation& op : body->without_terminator()) {
    if (!needed_ops.count(&op)) continue;
    builder.clone(op, mapping);
  }

  llvm::SmallVector<mlir::Value> new_yield_operands;
  new_yield_operands.reserve(live_yield_operands.size());
  for (mlir::Value operand : live_yield_operands) {
    new_yield_operands.push_back(mapping.lookupOrDefault(operand));
  }
  mlir::ktdf::PrivateYieldOp::create(builder, yield_op.getLoc(),
                                     new_yield_operands);

  for (auto [new_idx, old_idx] : llvm::enumerate(live_result_indices)) {
    private_op.getResult(old_idx).replaceAllUsesWith(
        new_private.getResult(new_idx));
  }

  private_op.erase();
  return mlir::success();
}

mlir::LogicalResult scheduler::cleanupPrivateOpsInPipeline(
    mlir::ktdf::PipelineOp pipeline_op) {
  const auto private_ops =
      llvm::to_vector(pipeline_op.getOps<mlir::ktdf::PrivateOp>());

  for (mlir::ktdf::PrivateOp private_op : private_ops) {
    if (mlir::failed(cleanupPrivateOp(private_op))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult scheduler::extractGridSize(mlir::func::FuncOp func,
                                               int& grid_size) {
  auto grid_attr = func->template getAttrOfType<mlir::ArrayAttr>("grid");
  if (!grid_attr) {
    return func.emitError("Function missing 'grid' attribute");
  }

  if (grid_attr.size() != 1) {
    return func.emitError("Only 1D grids supported in Phase 1 (got ")
           << grid_attr.size() << "D grid)";
  }

  auto size_attr = mlir::dyn_cast<mlir::IntegerAttr>(grid_attr[0]);
  if (!size_attr) {
    return func.emitError("Grid size must be integer attribute");
  }

  grid_size = size_attr.getInt();
  if (grid_size <= 0) {
    return func.emitError("Grid size must be positive (got ")
           << grid_size << ")";
  }

  return mlir::success();
}

mlir::scf::ForOp scheduler::createForOpWithAdditionalIterArgs(
    mlir::scf::ForOp loop_op, int n_values, mlir::IRMapping& ir_map,
    bool delete_op) {
  mlir::OpBuilder builder(loop_op);
  llvm::SmallVector<mlir::Value, 8> iter_args;

  // Copy existing iterator arguments
  for (auto operand : loop_op.getInits()) {
    iter_args.push_back(operand);
  }

  // Add new iterator arguments with dummy constant initial values
  for (int i = 0; i < n_values; i++) {
    auto const_op =
        mlir::arith::ConstantIndexOp::create(builder, loop_op.getLoc(), 0);
    iter_args.push_back(const_op);
  }

  // Create new loop with additional iter args
  auto new_loop = mlir::scf::ForOp::create(
      builder, loop_op.getLoc(), loop_op.getLowerBound(),
      loop_op.getUpperBound(), loop_op.getStep(), iter_args);

  // Copy attributes from old loop to new loop
  for (auto attr : loop_op->getAttrs()) {
    // Skip operandSegmentSizes as it will be automatically computed
    if (attr.getName() != "operandSegmentSizes") {
      new_loop->setAttr(attr.getName(), attr.getValue());
    }
  }

  // Clone the loop body
  builder.setInsertionPointToStart(new_loop.getBody());

  // Map the induction variable
  ir_map.map(loop_op.getInductionVar(), new_loop.getInductionVar());

  // Map existing region iter args
  for (auto [old_arg, new_arg] :
       llvm::zip(loop_op.getRegionIterArgs(), new_loop.getRegionIterArgs())) {
    ir_map.map(old_arg, new_arg);
  }

  // Clone operations in the loop body
  for (auto& op : loop_op.getBody()->without_terminator()) {
    builder.clone(op, ir_map);
  }

  // Create new yield operation
  llvm::SmallVector<mlir::Value> yield_operands;
  auto old_yield =
      mlir::cast<mlir::scf::YieldOp>(loop_op.getBody()->getTerminator());

  // Map existing yield operands
  for (auto operand : old_yield.getOperands()) {
    yield_operands.push_back(ir_map.lookup(operand));
  }

  // Add new yield operands (pass through the new iter args)
  for (size_t i = loop_op.getNumRegionIterArgs();
       i < new_loop.getNumRegionIterArgs(); i++) {
    yield_operands.push_back(new_loop.getRegionIterArgs()[i]);
  }

  mlir::scf::YieldOp::create(builder, old_yield.getLoc(), yield_operands);

  // Replace uses of old loop results with new loop results
  for (size_t i = 0; i < loop_op->getNumResults(); i++) {
    loop_op->getResult(i).replaceAllUsesWith(new_loop->getResult(i));
  }

  // Erase the old loop if requested
  if (delete_op) {
    loop_op->erase();
  }

  return new_loop;
}

void scheduler::eraseDeadAncestorOps(llvm::ArrayRef<mlir::Operation*> roots) {
  llvm::SmallVector<mlir::Operation*> worklist(roots);
  llvm::DenseSet<mlir::Operation*> erased;
  while (!worklist.empty()) {
    mlir::Operation* op = worklist.pop_back_val();
    // An op reached twice is on the worklist twice, and the second visit would
    // be to freed memory.
    if (erased.contains(op) || !mlir::isOpTriviallyDead(op)) continue;
    for (mlir::Value operand : op->getOperands())
      if (mlir::Operation* def = operand.getDefiningOp())
        worklist.push_back(def);
    erased.insert(op);
    op->erase();
  }
}
