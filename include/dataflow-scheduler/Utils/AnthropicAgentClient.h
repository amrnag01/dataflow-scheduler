//===-- AnthropicAgentClient.h -------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_UTILS_ANTHROPICAGENTCLIENT_H_
#define DATAFLOW_SCHEDULER_UTILS_ANTHROPICAGENTCLIENT_H_

#include <cstdint>
#include <memory>
#include <string>

#include "dataflow-scheduler/Dialect/KTDF/TileSizeInfo.h"
#include "mlir/IR/BuiltinOps.h"

namespace scheduler {

class AnthropicAgentClient {
public:
  explicit AnthropicAgentClient(const std::string& api_key);
  ~AnthropicAgentClient();

  int64_t selectTileSize(
      mlir::ModuleOp module,
      TileSizeInfo& tile_size_info);

private:
  std::string api_key_;

  std::string buildPrompt(
      mlir::ModuleOp module,
      TileSizeInfo& tile_size_info);

  std::string buildTaskDefinition();

  std::string buildConstraintsSection(
      TileSizeInfo& tile_size_info,
      int64_t min_value,
      int64_t divisibility);

  std::string buildContextSection(
      mlir::ModuleOp module,
      TileSizeInfo& tile_size_info);

  std::string buildHeuristicBaselineSection();

  std::string buildOutputFormatSection(int64_t min_value);

  std::string makeHttpRequest(const std::string& prompt);

  int64_t parseJsonResponse(const std::string& response);
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_UTILS_ANTHROPICAGENTCLIENT_H_
