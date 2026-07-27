//===-- dataflow-scheduler.cpp ----------------------------------*- c++ -*-===//
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
// This is a tool for consuming KTIR mlir and generating KTDF and/or DFIR.
//
//===----------------------------------------------------------------------===//

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <cstdlib>
#include <memory>

#include "dataflow-scheduler-main.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Pipeline.h"
#include "dataflow-scheduler/RegisterEverything.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"

// Static storage for the scheduler context to ensure it outlives pass execution
static std::unique_ptr<scheduler::SchedulerExtContext> g_scheduler_context;

void registerPassPipelinesForScheduler() {
  static llvm::cl::opt<std::string> splitDFIROutputDir(
      "split-dfir-output-dir",
      llvm::cl::desc("Output directory for split DFIR files produced by "
                     "-kEmitDFIR (default: same directory as input file)"),
      llvm::cl::init(""));

  static llvm::cl::opt<std::string> anthropicApiKey(
      "anthropic-api-key",
      llvm::cl::desc("Anthropic API key for agent-driven tile size selection"),
      llvm::cl::init(""));

  mlir::PassPipelineRegistration<>(
      "kEmitDFIR", "Emit DataflowIR", [&](mlir::OpPassManager& pm) {
        // Try to get API key from environment or CLI flag
        std::string api_key = anthropicApiKey;
        if (api_key.empty()) {
          const char* env_key = std::getenv("ANTHROPIC_API_KEY");
          if (env_key) {
            api_key = env_key;
          }
        }

        if (!api_key.empty()) {
          g_scheduler_context =
              std::make_unique<scheduler::AgentDrivenSchedulerContext>(api_key);
        } else {
          g_scheduler_context =
              std::make_unique<scheduler::DummySchedulerExtContext>();
          llvm::errs() << "Warning: No Anthropic API key provided. "
                          "Set ANTHROPIC_API_KEY environment variable or use "
                          "--anthropic-api-key flag.\n";
        }

        scheduler::buildKTDPToDFIRPipeline(
            pm, scheduler::SchedulerExtContext::dummyContext());
        pm.addPass(scheduler::createSplitDFIROutputPass(splitDFIROutputDir));
      });
}

// FIXME: We should use dataflow-scheduler-opt for internal testing, and turn
//        this executable into a self-contained tool front-end, without the
//        default MLIR CLI etc.
auto main(int argc, char** argv) -> int {
  scheduler::registerAllPasses();
  registerPassPipelinesForScheduler();

  DialectRegistry registry;
  scheduler::registerAllDialects(registry);
  scheduler::registerAllExtensions(registry);

  return asMainReturnCode(scheduler::SchedulerOptMain(
      argc, argv, "DataflowScheduler modular optimizer driver", registry));
}
