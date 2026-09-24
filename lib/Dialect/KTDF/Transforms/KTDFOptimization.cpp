//===-- KTDFOptimization.cpp -----------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"

#include <memory>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Utils/KTDFOptimizationAgent.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "ktdf-optimization"
#define DEBUG_TYPE PASS_NAME

namespace mlir::ktdf {
#define GEN_PASS_DEF_KTDFOPTIMIZATIONPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"

namespace {

struct KTDFOptimizationPass
    : public impl::KTDFOptimizationPassBase<KTDFOptimizationPass> {
  const scheduler::SchedulerExtContext* scheduler_ctx = nullptr;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::errs() << "[KTDFOptimization] Running optimization\n";

    std::string api_key = anthropicApiKey;

    // If no API key from pass option, try to get from context
    if (api_key.empty() && scheduler_ctx && !scheduler_ctx->isDummy()) {
      auto agent_ctx = static_cast<const scheduler::AgentDrivenSchedulerContext*>(scheduler_ctx);
      api_key = agent_ctx->api_key;
    }

    if (api_key.empty()) {
      llvm::errs() << "[KTDFOptimization] No API key provided, skipping agent "
                      "call\n";
      return;
    }

    // Get context paths
    std::string ktdf_bindings_dir;
    std::string cost_model_path;
    if (scheduler_ctx && !scheduler_ctx->isDummy()) {
      auto agent_ctx = static_cast<const scheduler::AgentDrivenSchedulerContext*>(scheduler_ctx);
      ktdf_bindings_dir = agent_ctx->ktdf_bindings_dir;
      cost_model_path = agent_ctx->cost_model_path;
    }

    auto agent = std::make_unique<scheduler::KTDFOptimizationAgent>(
        api_key, ktdf_bindings_dir, cost_model_path);
    mlir::ModuleOp optimized = agent->optimizeKTDF(module);

    if (optimized && optimized != module) {
      // Replace the original module's body with the optimized one
      OpBuilder builder(module.getContext());
      module.getBody()->clear();
      for (auto& op : optimized.getOps()) {
        module.getBody()->push_back(op.clone());
      }
      llvm::errs() << "[KTDFOptimization] Successfully replaced module with optimized IR\n";
    }

    llvm::errs() << "[KTDFOptimization] Optimization complete\n";
  }
};

}  // namespace

std::unique_ptr<Pass> createKTDFOptimizationPass() {
  return std::make_unique<KTDFOptimizationPass>();
}

std::unique_ptr<Pass> createKTDFOptimizationPass(
    const scheduler::SchedulerExtContext& scheduler_ctx) {
  auto pass = std::make_unique<KTDFOptimizationPass>();
  pass->scheduler_ctx = &scheduler_ctx;
  return pass;
}

}  // namespace mlir::ktdf
