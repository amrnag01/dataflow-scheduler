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
// Split DFIR output into multiple files for backend compilation.
//
// This pass takes a module containing function declarations and definitions
// with dataflow.program_unit ops, and splits it into:
//   - global.mlir: All function declarations
//   - <func-name>.mlir: Each function definition containing program_unit ops
//
// Example:
//   Input module with:
//     func.func @decl1() -> ()
//     func.func @"local-schedule-0"() { dataflow.program_unit ... }
//     func.func @"local-schedule-1"() { dataflow.program_unit ... }
//
//   Output:
//     global.mlir: @decl1 declaration
//     local-schedule-0.mlir: @"local-schedule-0" definition
//     local-schedule-1.mlir: @"local-schedule-1" definition
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "split-dfir-output"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_SPLITDFIROUTPUTPASS
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"
}  // namespace scheduler

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Split DFIR Output pass"),
    llvm::cl::init(false));

namespace {

struct SplitDFIROutputPass
    : public impl::SplitDFIROutputPassBase<SplitDFIROutputPass> {
  SplitDFIROutputPass() = default;
  explicit SplitDFIROutputPass(llvm::StringRef dir) { outputDir = dir.str(); }

  void runOnOperation() override {
    if (DisableThisPass) return;
    mlir::ModuleOp top_module = getOperation();

    // Validate: exactly two child modules required.
    llvm::SmallVector<mlir::ModuleOp, 2> child_modules;
    for (auto& op : top_module.getBodyRegion().front()) {
      if (auto mod = mlir::dyn_cast<mlir::ModuleOp>(op))
        child_modules.push_back(mod);
    }
    if (child_modules.size() != 2) {
      top_module.emitError()
          << "SplitDFIROutputPass expects exactly 2 child modules, got "
          << child_modules.size();
      signalPassFailure();
      return;
    }

    mlir::ModuleOp decl_module = child_modules[0];
    mlir::ModuleOp impl_module = child_modules[1];

    // Resolve output directory.
    llvm::SmallString<128> out_dir(outputDir);
    if (out_dir.empty()) {
      if (auto file_loc =
              mlir::dyn_cast<mlir::FileLineColLoc>(top_module.getLoc())) {
        llvm::StringRef parent =
            llvm::sys::path::parent_path(file_loc.getFilename().getValue());
        out_dir.assign(parent);
      } else {
        top_module.emitError()
            << "output-dir not specified and source location unavailable";
        signalPassFailure();
        return;
      }
    }

    // Create output directory.
    if (std::error_code ec = llvm::sys::fs::create_directories(
            out_dir, /*IgnoreExisting=*/true)) {
      top_module.emitError() << "failed to create output directory " << out_dir
                             << ": " << ec.message();
      signalPassFailure();
      return;
    }

    mlir::MLIRContext* ctx = top_module.getContext();
    mlir::OpBuilder builder(ctx);

    // Write a module to a file under out_dir.
    auto writeModule = [&](mlir::ModuleOp mod,
                           llvm::StringRef filename) -> bool {
      llvm::SmallString<128> path(out_dir);
      llvm::sys::path::append(path, filename);
      std::error_code ec;
      llvm::raw_fd_ostream os(path, ec);
      if (ec) {
        top_module.emitError()
            << "failed to open output file " << path << ": " << ec.message();
        return false;
      }
      mlir::OpPrintingFlags flags;
      flags.enableDebugInfo(false);
      mod.print(os, flags);
      os << "\n";
      return true;
    };

    // Derive filename from symbol name: @"local-schedule-0" ->
    // local-schedule-0.mlir
    auto funcNameToFilename = [](llvm::StringRef sym_name) -> std::string {
      if (sym_name.starts_with("\"") && sym_name.ends_with("\""))
        sym_name = sym_name.substr(1, sym_name.size() - 2);
      return (sym_name + ".mlir").str();
    };

    // True if func body contains any dataflow.program_unit op.
    auto hasProgramUnits = [](mlir::func::FuncOp func) -> bool {
      bool found = false;
      func.walk([&](mlir::dataflow::ProgramUnitOp) {
        found = true;
        return mlir::WalkResult::interrupt();
      });
      return found;
    };

    // Build global.mlir from decl_module.
    {
      mlir::OwningOpRef<mlir::ModuleOp> global_mod =
          mlir::ModuleOp::create(builder.getUnknownLoc());
      mlir::OpBuilder gb(global_mod->getBodyRegion());
      for (auto& op : decl_module.getBodyRegion().front()) {
        if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
          auto* cloned = gb.clone(op);
          if (auto f = mlir::dyn_cast<mlir::func::FuncOp>(cloned))
            f.setVisibility(mlir::SymbolTable::Visibility::Public);
        }
      }
      if (!writeModule(global_mod.get(), "global.mlir")) {
        signalPassFailure();
        return;
      }
    }

    // Build <funcname>.mlir for each impl function with program_unit ops.
    bool found_impl = false;
    bool write_failed = false;
    impl_module.walk([&](mlir::func::FuncOp func) {
      if (!hasProgramUnits(func)) return mlir::WalkResult::advance();
      found_impl = true;

      mlir::OwningOpRef<mlir::ModuleOp> impl_mod =
          mlir::ModuleOp::create(builder.getUnknownLoc());
      mlir::OpBuilder ib(impl_mod->getBodyRegion());
      auto* cloned = ib.clone(*func.getOperation());
      if (auto f = mlir::dyn_cast<mlir::func::FuncOp>(cloned))
        f.setVisibility(mlir::SymbolTable::Visibility::Public);

      std::string filename = funcNameToFilename(func.getSymName());
      if (!writeModule(impl_mod.get(), filename)) {
        write_failed = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (write_failed) {
      signalPassFailure();
      return;
    }

    if (!found_impl) {
      top_module.emitWarning()
          << "SplitDFIROutputPass: no implementation functions with "
             "dataflow.program_unit ops found";
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createSplitDFIROutputPass() {
  return std::make_unique<SplitDFIROutputPass>();
}

std::unique_ptr<mlir::Pass> scheduler::createSplitDFIROutputPass(
    llvm::StringRef output_dir) {
  return std::make_unique<SplitDFIROutputPass>(output_dir);
}
