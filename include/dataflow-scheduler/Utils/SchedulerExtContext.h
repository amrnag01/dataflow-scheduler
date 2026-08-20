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

#ifndef DATAFLOW_SCHEDULER_UTILS_SCHEDULEREXTCONTEXT_H_
#define DATAFLOW_SCHEDULER_UTILS_SCHEDULEREXTCONTEXT_H_

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>

#include "llvm/Support/Error.h"
#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDF/TileSizeInfo.h"

namespace scheduler {

// Forward declarations
class AnthropicAgentClient;
class AgenticTileSizeSelector;

// Type alias for resource representation
using ResourceType = mlir::Attribute;

struct SchedulerExtContext {
  SchedulerExtContext();
  virtual ~SchedulerExtContext();

  /// @brief Get a default initialized context for scheduler.
  /// The dummy context is intended for use with passes and analyses that
  /// require default constructors.
  static const SchedulerExtContext& dummyContext();

  virtual bool isDummy() const { return false; }

  /// @brief Select a tile size for the given tile size problem.
  /// Must be overridden by contexts that support agent-driven optimization.
  virtual int64_t selectTileSize(
      mlir::ModuleOp module,
      TileSizeInfo& tile_size_info) {
    llvm::report_fatal_error(
        "selectTileSize not implemented for this context");
  }

  /// @brief Select all tile sizes for a module at once using agentic loop.
  /// Must be overridden by contexts that support agent-driven optimization.
  virtual std::vector<int64_t> selectAllTileSizes(
      mlir::ModuleOp module,
      llvm::ArrayRef<TileSizeInfo> analyses) {
    llvm::report_fatal_error(
        "selectAllTileSizes not implemented for this context");
  }
};

/// @brief Construct a DummySchedulerExtContext. In general prefer
/// SchedulerExtContext::dummyContext.
struct DummySchedulerExtContext : SchedulerExtContext {
  DummySchedulerExtContext() : SchedulerExtContext() {}
  bool isDummy() const override { return true; }
};

/// @brief Context with agent-driven tile size optimization.
struct AgentDrivenSchedulerContext : SchedulerExtContext {
  std::unique_ptr<AnthropicAgentClient> agent_client;
  std::string ktdf_bindings_dir;
  std::string cost_model_path;
  std::string api_key;
  bool debug;

  AgentDrivenSchedulerContext(
      const std::string& api_key,
      const std::string& ktdf_bindings_dir,
      const std::string& cost_model_path,
      bool debug = false);
  ~AgentDrivenSchedulerContext();

  virtual int64_t selectTileSize(
      mlir::ModuleOp module,
      TileSizeInfo& tile_size_info) override;

  virtual std::vector<int64_t> selectAllTileSizes(
      mlir::ModuleOp module,
      llvm::ArrayRef<TileSizeInfo> analyses) override;
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_UTILS_SCHEDULEREXTCONTEXT_H_
