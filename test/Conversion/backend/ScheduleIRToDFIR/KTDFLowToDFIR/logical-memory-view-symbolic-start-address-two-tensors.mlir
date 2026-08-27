// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Two tensors the compiler placed at addresses of its own, each displaced by the
// same row the run hands in. What each grid element reads is a different address
// in each tensor, so each needs a symbol of its own, and what tells them apart is
// the expression: the input, the offset and the word size are the same for both.
//
// One symbol for two addresses would have the run resolve one of them and the
// other read whatever the first landed at.

// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index) attributes {grid = [2]} {
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[C128000:.*]] = arith.constant 128000 : index
// CHECK:           %[[C64000:.*]] = arith.constant 64000 : index
// CHECK:           %[[C4096:.*]] = arith.constant 4096 : index
// CHECK:           %[[C64:.*]] = arith.constant 64 : index
// CHECK:           symbol.create_id %arg0 {symbol_id = -1 : si64} : index

// The first tensor: the row the run hands in times the rows it strides, plus the
// base the compiler picked, in bytes, over the word size.
// CHECK-NEXT:      %[[ROWS_A:.*]] = arith.muli %arg0, %[[C4096]] : index
// CHECK-NEXT:      symbol.create_id %[[ROWS_A]] {symbol_id = -2 : si64} : index
// CHECK-NEXT:      %[[ELEMS_A:.*]] = arith.addi %[[ROWS_A]], %[[C64000]] : index
// CHECK-NEXT:      symbol.create_id %[[ELEMS_A]] {symbol_id = -3 : si64} : index
// CHECK-NEXT:      %[[BYTES_A:.*]] = arith.muli %[[ELEMS_A]], %[[C2]] : index
// CHECK-NEXT:      symbol.create_id %[[BYTES_A]] {symbol_id = -4 : si64} : index
// CHECK-NEXT:      %[[WORDS_A:.*]] = arith.divsi %[[BYTES_A]], %[[C64]] : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_A]] {symbol_id = -5 : si64} : index

// And the second. The displacement is the same one, so it is the same symbol --
// -2 again -- said over this address's own rebuild of it, which is the value the
// step above it reads. What differs is the base, and from there the ids are new.
// CHECK-NEXT:      %[[ROWS_B:.*]] = arith.muli %arg0, %[[C4096]] : index
// CHECK-NEXT:      symbol.create_id %[[ROWS_B]] {symbol_id = -2 : si64} : index
// CHECK-NEXT:      %[[ELEMS_B:.*]] = arith.addi %[[ROWS_B]], %[[C128000]] : index
// CHECK-NEXT:      symbol.create_id %[[ELEMS_B]] {symbol_id = -6 : si64} : index
// CHECK-NEXT:      %[[BYTES_B:.*]] = arith.muli %[[ELEMS_B]], %[[C2]] : index
// CHECK-NEXT:      symbol.create_id %[[BYTES_B]] {symbol_id = -7 : si64} : index
// CHECK-NEXT:      %[[WORDS_B:.*]] = arith.divsi %[[BYTES_B]], %[[C64]] : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_B]] {symbol_id = -8 : si64} : index

// And the two views, each reading its own.
// CHECK:           %[[SYM_A:.*]] = symbol.create_symbol {SymbolId = -5 : i64} : index
// CHECK:           dataflow.get_logical_memory_view %{{.*}}, %[[SYM_A]]
// CHECK:           %[[SYM_B:.*]] = symbol.create_symbol {SymbolId = -8 : i64} : index
// CHECK:           dataflow.get_logical_memory_view %{{.*}}, %[[SYM_B]]

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%row_0: index) attributes {grid = [2]} {
      call @sched_0(%row_0) : (index) -> ()
      return
    }
    func.func private @sched_0(index)
  }
  module {
    func.func private @sched_0(%row: index) attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %c4096 = arith.constant 4096 : index
      %c64000 = arith.constant 64000 : index
      %c128000 = arith.constant 128000 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        %shift = arith.muli %row, %c4096 : index

        %view_a = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc_a = memref.memory_space_cast %view_a : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc_a = memref.reinterpret_cast %msc_a to offset: [%shift], sizes: [6, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<6x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1_a = builtin.unrealized_conversion_cast %c128 : index to memref<6x1x64x64xf16, "L1">

        %view_b = ktdp.construct_memory_view %c128000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc_b = memref.memory_space_cast %view_b : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc_b = memref.reinterpret_cast %msc_b to offset: [%shift], sizes: [6, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<6x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1_b = builtin.unrealized_conversion_cast %c256 : index to memref<6x1x64x64xf16, "L1">

        scf.for %arg1 = %c0 to %c12 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc_a[%arg1, %c0, %arg2, %c0] size [1, 1, 1, 64] to %l1_a[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] : memref<6x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<6x1x64x64xf16, "L1">
            ktdf.data_transfer from %rc_b[%arg1, %c0, %arg2, %c0] size [1, 1, 1, 64] to %l1_b[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] : memref<6x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<6x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1_a : memref<6x1x64x64xf16, "L1">
        memref.dealloc %l1_b : memref<6x1x64x64xf16, "L1">
      }
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%6, %7) : {
      }
      return
    }
  }
}
