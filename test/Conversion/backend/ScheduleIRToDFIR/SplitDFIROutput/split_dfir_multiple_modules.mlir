// RUN: rm -rf %t && mkdir -p %t
// RUN: dataflow-scheduler-opt -allow-unregistered-dialect --emit-split-dfir='output-dir=%t' %s 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: FileCheck %s --input-file=%t/global.mlir --check-prefix=GLOBAL
// RUN: FileCheck %s --input-file=%t/local_schedule_0_idx_to_addr_body.mlir --check-prefix=IMPL0
// RUN: FileCheck %s --input-file=%t/local_schedule_0_body.mlir --check-prefix=IMPL1

// WARN-NOT: error
// WARN-NOT: SplitDFIROutputPass: no implementation functions

// GLOBAL-LABEL: "builtin.module"() ({
// GLOBAL:         "func.func"() {{.*}} sym_name = "my_func"
// GLOBAL:         "func.func"() {{.*}} sym_name = "local_schedule_0"
// GLOBAL:         "func.func"() {{.*}} sym_name = "local_schedule_0_idx_to_addr"
// GLOBAL-NOT:     dataflow.program_unit
// GLOBAL-NOT:     ktdf_arch.device

// IMPL0-LABEL: module {
// IMPL0:         func.func @local_schedule_0_idx_to_addr_body
// IMPL0:         dataflow.program_unit
// IMPL0:           "test.idx_to_addr"
// IMPL0-NOT:     func.func @local_schedule_0_body

// IMPL1-LABEL: module {
// IMPL1:         func.func @local_schedule_0_body
// IMPL1:         dataflow.program_unit
// IMPL1:           "test.schedule"
// IMPL1-NOT:     func.func @local_schedule_0_idx_to_addr_body

// Tests that SplitDFIROutputPass handles one implementation module per
// schedule, as produced by wrap-program-dfir, with other top-level ops (here a
// device) between them.

module {
  module {
    func.func @my_func() attributes {grid = [1]} {
      call @local_schedule_0_idx_to_addr() : () -> ()
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
    func.func private @local_schedule_0_idx_to_addr()
  }
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0_idx_to_addr {
    func.func private @local_schedule_0_idx_to_addr_body()
    func.func @local_schedule_0_idx_to_addr() {
      call @local_schedule_0_idx_to_addr_body() : () -> ()
      return
    }
    module {
      func.func @local_schedule_0_idx_to_addr_body() attributes {grid = [1]} {
        %0 = dataflow.get_unit {name = "DDR", type = "DDR"} : index
        dataflow.program_unit iter_arg : %arg0 -> (%0) : {
          "test.idx_to_addr"() : () -> ()
        }
        return
      }
    }
  }
  module @local_schedule_0 {
    func.func private @local_schedule_0_body()
    func.func @local_schedule_0() {
      call @local_schedule_0_body() : () -> ()
      return
    }
    module {
      func.func @local_schedule_0_body() attributes {grid = [1]} {
        %0 = dataflow.get_unit {name = "DDR", type = "DDR"} : index
        dataflow.program_unit iter_arg : %arg0 -> (%0) : {
          "test.schedule"() : () -> ()
        }
        return
      }
    }
  }
}
