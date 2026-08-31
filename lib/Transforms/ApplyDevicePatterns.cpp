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

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "dataflow-scheduler/Dialect/Agen/Agen.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchDialect.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDFArch/Transforms/ApplyPatterns.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Transforms/Passes.h"

#define PASS_NAME "apply-device-patterns"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Device Patterns pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_APPLYDEVICEPATTERNSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

class PatternCache : public mlir::ktdf_arch::PatternCache {
 public:
  using mlir::ktdf_arch::PatternCache::PatternCache;

  void registerNativeFunctions(mlir::PDLPatternModule& patterns) final {
    mlir::ktdf_arch::PatternCache::registerNativeFunctions(patterns);

    // TODO: Register additional native functions.
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PatternCache)
};

struct ApplyDevicePatternsPass
    : impl::ApplyDevicePatternsPassBase<ApplyDevicePatternsPass> {
  using ApplyDevicePatternsPassBase::ApplyDevicePatternsPassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    // Find the device that this function maps to.
    auto declaration =
        mlir::ktdf_arch::findDeviceDeclarationFor(getOperation());
    if (!declaration) {
      return;
    }
    mlir::ktdf_arch::DeviceRef device(declaration, getAnalysisManager());
    LDBG() << "processing "
           << mlir::OpWithFlags(getOperation(),
                                mlir::OpPrintingFlags().skipRegions())
           << " with device " << declaration.getName();

    // Get the (cached) rewrite pattern set. This prevents cloning the PDL
    // module.
    const auto patterns = device.getOrCreateView<PatternCache>().get(
        mlir::ktdf_arch::PatternGroups(llvm::from_range, enabled_groups));

    // Run all the patterns.
    auto changed = false;
    if (failed(applyPatternsGreedily(getOperation(), patterns,
                                     mlir::GreedyRewriteConfig(), &changed))) {
      signalPassFailure();
      return;
    }

    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};

}  // namespace

auto scheduler::createApplyDevicePatternsPass(
    std::initializer_list<llvm::StringRef> enabled_groups)
    -> std::unique_ptr<mlir::Pass> {
  ApplyDevicePatternsPassOptions options;
  for (auto group : enabled_groups) {
    options.enabled_groups.emplace_back(group.str());
  }
  return createApplyDevicePatternsPass(options);
}
