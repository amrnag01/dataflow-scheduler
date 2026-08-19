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

#ifndef DATAFLOW_SCHEDULER_UTILS_AGENTICTILESZESELECTOR_H_
#define DATAFLOW_SCHEDULER_UTILS_AGENTICTILESZESELECTOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#include <nlohmann/json.hpp>
#pragma clang diagnostic pop

#include "dataflow-scheduler/Dialect/KTDF/TileSizeInfo.h"
#include "mlir/IR/BuiltinOps.h"

namespace scheduler {

// Agentic tile size selector using tool-use loop with Claude
class AgenticTileSizeSelector {
 public:
  explicit AgenticTileSizeSelector(
      const std::string& api_key,
      const std::string& ktdf_bindings_dir,
      const std::string& cost_model_path);
  ~AgenticTileSizeSelector();

  // Run the tool-use loop to select tile sizes for all reserve_size ops
  // Returns a vector of tile sizes, one per TileSizeInfo in analyses
  std::vector<int64_t> run(
      mlir::ModuleOp module,
      llvm::ArrayRef<TileSizeInfo> analyses);

 private:
  std::string api_key_;
  std::string ktdf_bindings_dir_;
  std::string cost_model_path_;

  // Prompt building
  std::string buildSystemPrompt(llvm::ArrayRef<TileSizeInfo> analyses);
  std::string buildToolSchemas();

  // Tool execution
  struct TransformResult {
    bool success;
    double latency;  // only valid if success
    std::string error_message;
  };

  TransformResult handleTransformAndEvaluateCost(
      mlir::ModuleOp module,
      llvm::ArrayRef<TileSizeInfo> analyses,
      const std::vector<std::pair<int64_t, int64_t>>& tile_size_assignments);

  // Cost model subprocess
  TransformResult runCostModelSubprocess(
      const std::string& ir_file,
      const std::string& ir_str,
      const std::vector<std::pair<int64_t, int64_t>>& tile_size_assignments);

  // HTTP communication with Claude
  std::string makeHttpRequestWithTools(
      const std::string& system_prompt,
      const nlohmann::json& messages,
      const std::string& tool_schemas);
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_UTILS_AGENTICTILESZESELECTOR_H_
