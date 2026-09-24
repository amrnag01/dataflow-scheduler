//===-- KTDFOptimizationAgent.h -------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_UTILS_KTDFOPTIMIZATIONAGENT_H_
#define DATAFLOW_SCHEDULER_UTILS_KTDFOPTIMIZATIONAGENT_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"

namespace scheduler {

class KTDFOptimizationAgent {
public:
  explicit KTDFOptimizationAgent(
      const std::string& api_key,
      const std::string& ktdf_bindings_dir,
      const std::string& cost_model_path);
  ~KTDFOptimizationAgent();

  /// Optimize the given module. Returns the optimized ModuleOp from the agent.
  mlir::ModuleOp optimizeKTDF(mlir::ModuleOp module);

  /// Get path to optimized IR temp file (if one was created)
  std::string getOptimizedIRPath() const { return optimized_ir_path_; }

private:
  std::string api_key_;
  std::string ktdf_bindings_dir_;
  std::string cost_model_path_;
  std::string optimized_ir_path_;

  struct CostEvaluation {
    bool success;
    double latency;
    std::map<std::string, double> variables;
    std::string error_message;
  };

  std::string makeHttpRequest(const std::string& prompt);

  std::string buildSystemPrompt();
  std::string buildToolSchemas();

  CostEvaluation evaluateCost(mlir::ModuleOp module);
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_UTILS_KTDFOPTIMIZATIONAGENT_H_
