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
// Path Expansion Pass
//
// This pass legalizes data movement in ktdf.pipeline operations using an
// explicit architecture graph. It expands illegal direct transfers into
// legal multi-hop paths by inserting intermediate stages and resources.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"
#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/PathExpansion/Materializer.h"
#include "dataflow-scheduler/Transforms/PathExpansion/Planner.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "path-expansion"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Path Expansion pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_PATHEXPANSIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

//===----------------------------------------------------------------------===//
// Path Expansion Pass
//===----------------------------------------------------------------------===//

struct PathExpansionPass
    : public impl::PathExpansionPassBase<PathExpansionPass> {
  PathExpansionPass() : scheduler_ctx_(SchedulerExtContext::dummyContext()) {}
  explicit PathExpansionPass(const SchedulerExtContext& ctx)
      : scheduler_ctx_(ctx) {}

  void runOnOperation() override;

 private:
  /// Process a single pipeline operation
  mlir::LogicalResult processPipeline(
      mlir::ktdf::PipelineOp pipeline_op,
      const scheduler::arch_view::RoutingGraph& routing_graph);

  const SchedulerExtContext& scheduler_ctx_;
};

void PathExpansionPass::runOnOperation() {
  if (DisableThisPass) return;
  LDBG(1) << "========= " PASS_NAME " =========";

  mlir::ModuleOp module_op = getOperation();

  auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
  auto* const device = device_manager.getOrImportDevice();
  if (!device) {
    module_op->emitError(
        "Unable to import the device specification for path expansion. This "
        "could happen if the device spec file is empty or contains multiple "
        "devices");
    signalPassFailure();
    return;
  }
  auto& routing_graph =
      device_manager.getOrCreateView<arch_view::RoutingGraph>(*device);

  LLVM_DEBUG({
    llvm::dbgs() << "Routing Graph:\n";
    routing_graph.dump();
  });

  // Walk all pipeline operations in the module
  llvm::SmallVector<mlir::ktdf::PipelineOp> candidates;
  module_op.walk<mlir::WalkOrder::PostOrder>(
      [&](mlir::ktdf::PipelineOp pipeline_op) {
        candidates.push_back(pipeline_op);
      });
  for (mlir::ktdf::PipelineOp cand : candidates) {
    if (mlir::failed(processPipeline(cand, routing_graph))) {
      signalPassFailure();
      return;
    }
  }
}

mlir::LogicalResult PathExpansionPass::processPipeline(
    mlir::ktdf::PipelineOp pipeline_op,
    const scheduler::arch_view::RoutingGraph& routing_graph) {
  LDBG(1) << "Processing pipeline at " << pipeline_op.getLoc() << "";

  // Build PipelineTree for this pipeline
  PipelineTree tree;
  mlir::Operation* parent = pipeline_op->getParentOp();
  if (!parent) {
    pipeline_op->emitError("path-expansion: pipeline has no parent operation");
    return mlir::failure();
  }

  tree.compute(*parent);

  // Find the pipeline node in the tree
  PipelineNode* pipeline_node =
      static_cast<PipelineNode*>(tree.getNodeForOp(pipeline_op));

  // Plan path expansion
  auto plan = planPathExpansion(tree, pipeline_node, routing_graph);
  if (!plan) {
    pipeline_op->emitError("path-expansion: failed to plan path expansion");
    return mlir::failure();
  }

  if (!plan->changed) {
    LDBG(1) << "  Pipeline already legal, no changes needed";
    return mlir::success();
  }

  LDBG(1) << "  Pipeline needs expansion";

  // Materialize the modified pipeline tree
  mlir::OpBuilder builder(pipeline_op.getContext());
  mlir::IRMapping value_map;

  PathExpansionMaterializer materializer(builder, value_map, plan->stage_info,
                                         plan->resource_factory);

  // Set insertion point before the original pipeline
  builder.setInsertionPoint(pipeline_op);

  // Materialize the new pipeline from the modified tree
  mlir::Operation* new_pipeline_op = materializer.materialize(*pipeline_node);

  if (!new_pipeline_op) {
    pipeline_op->emitError(
        "path-expansion: failed to materialize expanded pipeline");
    return mlir::failure();
  }

  LDBG(1) << "  Materialization succeeded";

  auto new_pipeline = mlir::dyn_cast<mlir::ktdf::PipelineOp>(new_pipeline_op);
  if (!new_pipeline) {
    new_pipeline_op->emitError(
        "path-expansion: materialized operation is not a pipeline");
    return mlir::failure();
  }

  mlir::IRRewriter rewriter(new_pipeline);
  mlir::ktdf::PrivateBuilder::canonicalize(rewriter, new_pipeline);

  // Erase the original pipeline
  pipeline_op.erase();

  return mlir::success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> scheduler::createPathExpansionPass() {
  return std::make_unique<PathExpansionPass>();
}

std::unique_ptr<mlir::Pass> scheduler::createPathExpansionPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<PathExpansionPass>(scheduler_ctx);
}

// Made with Bob
