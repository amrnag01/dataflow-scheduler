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

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir::ktdf {

class KTDFOptimizationPass : public PassWrapper<KTDFOptimizationPass, OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final { return "ktdf-optimization"; }
  StringRef getDescription() const final {
    return "Placeholder pass for KTDF optimizations";
  }

  void runOnOperation() final {
    // Placeholder: no-op, just returns
  }
};

std::unique_ptr<Pass> createKTDFOptimizationPass() {
  return std::make_unique<KTDFOptimizationPass>();
}

}  // namespace mlir::ktdf
