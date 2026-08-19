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

#include "dataflow-scheduler/Utils/SchedulerExtContext.h"

#include <mlir/IR/Attributes.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Utils/AgenticTileSizeSelector.h"
#include "dataflow-scheduler/Utils/AnthropicAgentClient.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "mlir/IR/Builders.h"

using namespace scheduler;

SchedulerExtContext::SchedulerExtContext() = default;

SchedulerExtContext::~SchedulerExtContext() = default;

const SchedulerExtContext& SchedulerExtContext::dummyContext() {
  // The dummy context is initialized thread-safe and never written to.
  static const DummySchedulerExtContext dummy_ctx;
  return dummy_ctx;
}

AgentDrivenSchedulerContext::AgentDrivenSchedulerContext(
    const std::string& api_key,
    const std::string& ktdf_bindings_dir,
    const std::string& cost_model_path)
    : agent_client(std::make_unique<AnthropicAgentClient>(api_key)),
      ktdf_bindings_dir(ktdf_bindings_dir),
      cost_model_path(cost_model_path),
      api_key(api_key) {}

AgentDrivenSchedulerContext::~AgentDrivenSchedulerContext() = default;

int64_t AgentDrivenSchedulerContext::selectTileSize(
    mlir::ModuleOp module, TileSizeInfo& tile_size_info) {
  return agent_client->selectTileSize(module, tile_size_info);
}

std::vector<int64_t> AgentDrivenSchedulerContext::selectAllTileSizes(
    mlir::ModuleOp module,
    llvm::ArrayRef<TileSizeInfo> analyses) {
  AgenticTileSizeSelector selector(api_key, ktdf_bindings_dir, cost_model_path);
  return selector.run(module, analyses);
}
