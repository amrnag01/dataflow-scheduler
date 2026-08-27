// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// The other way a run input reaches an address: not as the start address of the
// view, which is a constant here, but through the offset of the tile read out of
// it. The kernel divides the tensor over the grid and then displaces each grid
// element's slab by a row the run hands in, so what varies is the offset and the
// view's own address is one the compiler picked.
//
// It is the same address either way, and it becomes a symbol the same way: the
// whole sum, in bytes, one symbol per grid element. Both halves of the sum count
// elements -- the base because the compiler counted it that way, the offset
// because it came off a memref -- and a symbol's definition is divided by the
// word size, which counts bytes, so the sum is scaled by the element size first.

// CHECK-LABEL:   func.func @test(
// CHECK-SAME:      %[[ROW:.*]]: index)
// CHECK-NEXT:      symbol.create_id %[[ROW]] {symbol_id = -1 : si64} : index

// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index) attributes {grid = [2]} {
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[C64000:.*]] = arith.constant 64000 : index
// CHECK:           %[[C4096:.*]] = arith.constant 4096 : index
// CHECK:           %[[C6:.*]] = arith.constant 6 : index
// CHECK:           %[[C64:.*]] = arith.constant 64 : index
// CHECK:           %[[DDR:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"}
// CHECK:           symbol.create_id %arg0 {symbol_id = -1 : si64} : index

// The first grid element, whose slab starts at the tensor's own base: the row the
// run hands in, times the rows it strides, plus the base, in bytes, over the word
// size. A step another step reads is a symbol of its own, because a definition is
// one operator over symbols and constants.
// CHECK-NEXT:      %[[ROWS_0:.*]] = arith.muli %arg0, %[[C4096]] : index
// CHECK-NEXT:      symbol.create_id %[[ROWS_0]] {symbol_id = -2 : si64} : index
// CHECK-NEXT:      %[[ELEMS_0:.*]] = arith.addi %[[ROWS_0]], %[[C64000]] : index
// CHECK-NEXT:      symbol.create_id %[[ELEMS_0]] {symbol_id = -3 : si64} : index
// CHECK-NEXT:      %[[BYTES_0:.*]] = arith.muli %[[ELEMS_0]], %[[C2]] : index
// CHECK-NEXT:      symbol.create_id %[[BYTES_0]] {symbol_id = -4 : si64} : index
// CHECK-NEXT:      %[[WORDS_0:.*]] = arith.divsi %[[BYTES_0]], %[[C64]] : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_0]] {symbol_id = -5 : si64} : index

// And the second, six rows further in. The six is the grid element's own and is a
// constant here; the row is not, which is why the sum needs a symbol at all.
// CHECK-NEXT:      %[[START_1:.*]] = arith.addi %arg0, %[[C6]] : index
// CHECK-NEXT:      symbol.create_id %[[START_1]] {symbol_id = -6 : si64} : index
// CHECK-NEXT:      %[[ROWS_1:.*]] = arith.muli %[[START_1]], %[[C4096]] : index
// CHECK-NEXT:      symbol.create_id %[[ROWS_1]] {symbol_id = -7 : si64} : index
// CHECK-NEXT:      %[[ELEMS_1:.*]] = arith.addi %[[ROWS_1]], %[[C64000]] : index
// CHECK-NEXT:      symbol.create_id %[[ELEMS_1]] {symbol_id = -8 : si64} : index
// CHECK-NEXT:      %[[BYTES_1:.*]] = arith.muli %[[ELEMS_1]], %[[C2]] : index
// CHECK-NEXT:      symbol.create_id %[[BYTES_1]] {symbol_id = -9 : si64} : index
// CHECK-NEXT:      %[[WORDS_1:.*]] = arith.divsi %[[BYTES_1]], %[[C64]] : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_1]] {symbol_id = -10 : si64} : index

// In the program, the symbol each grid element reads, chosen by a map keyed on the
// unit. Neither the offset arithmetic nor the reinterpret_cast it fed survives:
// the whole start address is the symbol, and arithmetic over the program's own
// argument left in the body would stop the program being split into two levels.
// CHECK:           %[[SYM_0:.*]] = symbol.create_symbol {SymbolId = -5 : i64} : index
// CHECK-NEXT:      %[[SYM_1:.*]] = symbol.create_symbol {SymbolId = -10 : i64} : index
// CHECK-NEXT:      %[[MAP:.*]] = uniform.def_immutable_mapping({{\[}}%{{.*}} -> %[[SYM_0]]], {{\[}}%{{.*}} -> %[[SYM_1]]]):index
// CHECK-NEXT:      %[[Q:.*]] = uniform.query_map(map:%[[MAP]], key:%{{.*}}) : index
// CHECK-NEXT:      dataflow.get_logical_memory_view %[[DDR]], %[[Q]]
// CHECK-NOT:       memref.reinterpret_cast

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
      %c6 = arith.constant 6 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      // Where the compiler put the tensor, rather than where the run did.
      %c64000 = arith.constant 64000 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        // What ktdp.get_compute_tile_id has become by this point: one constant
        // per core, chosen by the unit the program is running on.
        %tilemap = uniform.def_immutable_mapping([%0 -> %c0], [%1 -> %c1]):index
        %tile = uniform.query_map(map:%tilemap, key:%arg0) : index
        // Six rows per grid element, displaced by the row the run hands in.
        %start = arith.muli %tile, %c6 : index
        %rows = arith.addi %start, %row : index
        %shift = arith.muli %rows, %c4096 : index
        %10 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc = memref.memory_space_cast %10 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc = memref.reinterpret_cast %msc to offset: [%shift], sizes: [6, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<6x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<6x1x64x64xf16, "L1">
        scf.for %arg1 = %c0 to %c12 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc[%arg1, %c0, %arg2, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] : memref<6x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<6x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1 : memref<6x1x64x64xf16, "L1">
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
