// RUN: mkdir -p %t
// RUN: dataflow-scheduler-opt -allow-unregistered-dialect --emit-split-dfir='output-dir=%t' %s 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: FileCheck %s --input-file=%t/global.mlir --check-prefix=GLOBAL
// RUN: FileCheck %s --input-file=%t/local_schedule_0_body.mlir --check-prefix=IMPL

// WARN-NOT: SplitDFIROutputPass: no implementation functions

// GLOBAL-LABEL: "builtin.module"() ({
// GLOBAL:         "func.func"() {{.*}} sym_name = "my_func"
// GLOBAL:         "func.func"() {{.*}} sym_name = "local_schedule_0"
// GLOBAL-NOT:     dataflow.program_unit

// IMPL-LABEL: module {
// IMPL:           func.func @local_schedule_0_body
// IMPL:           dataflow.program_unit
// IMPL-NOT:       func.func @my_func

// Tests that SplitDFIROutputPass finds func.func ops with dataflow.program_unit
// ops nested inside an anonymous child module of the impl module, matching the
// structure produced by wrap-program-dfir:
//
//   module @local_schedule_0 {
//     func.func private @local_schedule_0_body()
//     func.func @local_schedule_0() { call @local_schedule_0_body() }
//     module {
//       func.func @local_schedule_0_body() { dataflow.program_unit ... }
//     }
//   }
//

module {
  module {
    func.func @my_func() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  module @local_schedule_0 {
    func.func private @local_schedule_0_body()
    func.func @local_schedule_0() attributes {grid = [1]} {
      call @local_schedule_0_body() : () -> ()
      return
    }
    module {
      func.func @local_schedule_0_body() attributes {grid = [1]} {
        %0 = dataflow.get_unit {name = "DDR", type = "DDR"} : index
        dataflow.program_unit iter_arg : %arg0 -> (%0) : {
          "test.op"() : () -> ()
        }
        return
      }
    }
  }
}
