// RUN: dataflow-scheduler-opt --normalize-scf-for-loops --canonicalize %s | FileCheck %s
// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 11 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>



// CHECK-LABEL:   func.func private @"local-schedule-0"() attributes {grid = [4 : index, 4 : index]} {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 15 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 786432 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_6:.*]] = arith.constant 162304 : index
// CHECK-NEXT:     %[[VAL_7:.*]] = arith.constant 113152 : index
// CHECK-NEXT:     %[[VAL_8:.*]] = arith.constant 192 : index
// CHECK-NEXT:     %[[VAL_9:.*]] = arith.constant 64000 : index
// CHECK-NEXT:     %[[VAL_10:.*]]:2 = ktdp.get_compute_tile_id : index, index
// CHECK-NEXT:     %[[VAL_11:.*]] = arith.muli %[[VAL_10]]#0, %[[VAL_8]] : index
// CHECK-NEXT:     %[[VAL_12:.*]] = ktdp.construct_memory_view %[[VAL_9]], sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = #ktdp.memory_space<global>} : memref<12x64x64xf16>
// CHECK-NEXT:     %[[VAL_13:.*]] = arith.muli %[[VAL_10]]#0, %[[VAL_3]] : index
// CHECK-NEXT:     %[[VAL_14:.*]] = arith.muli %[[VAL_10]]#1, %[[VAL_2]] : index
// CHECK-NEXT:     %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : index
// CHECK-NEXT:     %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_11]] : index
// CHECK-NEXT:     %[[VAL_17:.*]] = memref.memory_space_cast %[[VAL_12]] : memref<12x64x64xf16> to memref<12x64x64xf16, #ktdp.memory_space<global>>
// CHECK-NEXT:     %[[VAL_18:.*]] = memref.reinterpret_cast %[[VAL_17]] to offset: {{\[}}%[[VAL_16]]], sizes: [192, 16, 192], strides: [4096, 64, 1] : memref<12x64x64xf16, #ktdp.memory_space<global>> to memref<192x16x192xf16, strided<[4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:     %[[VAL_19:.*]] = ktdp.construct_memory_view %[[VAL_7]], sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = #ktdp.memory_space<global>} : memref<12x64x64xf16>
// CHECK-NEXT:     %[[VAL_20:.*]] = arith.muli %[[VAL_10]]#0, %[[VAL_3]] : index
// CHECK-NEXT:     %[[VAL_21:.*]] = arith.muli %[[VAL_10]]#1, %[[VAL_2]] : index
// CHECK-NEXT:     %[[VAL_22:.*]] = arith.addi %[[VAL_20]], %[[VAL_21]] : index
// CHECK-NEXT:     %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_11]] : index
// CHECK-NEXT:     %[[VAL_24:.*]] = memref.memory_space_cast %[[VAL_19]] : memref<12x64x64xf16> to memref<12x64x64xf16, #ktdp.memory_space<global>>
// CHECK-NEXT:     %[[VAL_25:.*]] = memref.reinterpret_cast %[[VAL_24]] to offset: {{\[}}%[[VAL_23]]], sizes: [192, 16, 192], strides: [4096, 64, 1] : memref<12x64x64xf16, #ktdp.memory_space<global>> to memref<192x16x192xf16, strided<[4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:     %[[VAL_26:.*]] = ktdp.construct_memory_view %[[VAL_6]], sizes: [1, 12, 64, 64], strides: [49152, 4096, 64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<1x12x64x64xf16>
// CHECK-NEXT:     %[[VAL_27:.*]] = arith.muli %[[VAL_10]]#0, %[[VAL_3]] : index
// CHECK-NEXT:     %[[VAL_28:.*]] = arith.muli %[[VAL_10]]#1, %[[VAL_2]] : index
// CHECK-NEXT:     %[[VAL_29:.*]] = arith.addi %[[VAL_27]], %[[VAL_28]] : index
// CHECK-NEXT:     %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_11]] : index
// CHECK-NEXT:     %[[VAL_31:.*]] = memref.memory_space_cast %[[VAL_26]] : memref<1x12x64x64xf16> to memref<1x12x64x64xf16, #ktdp.memory_space<global>>
// CHECK-NEXT:     %[[VAL_32:.*]] = memref.reinterpret_cast %[[VAL_31]] to offset: {{\[}}%[[VAL_30]]], sizes: [1, 192, 16, 192], strides: [49152, 4096, 64, 1] : memref<1x12x64x64xf16, #ktdp.memory_space<global>> to memref<1x192x16x192xf16, strided<[49152, 4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:     scf.for %[[VAL_33:.*]] = %[[VAL_4]] to %[[VAL_8]] step %[[VAL_5]] {
// CHECK-NEXT:       scf.for %[[VAL_34:.*]] = %[[VAL_4]] to %[[VAL_1]] step %[[VAL_5]] {
// CHECK-NEXT:         scf.for %[[VAL_35:.*]] = %[[VAL_4]] to %[[VAL_0]] step %[[VAL_5]] {
// CHECK-NEXT:           ktdf.pipeline {
// CHECK-NEXT:             %[[PRIV:.*]]:5 = ktdf.private -> (memref<1x1x64xf16, "L1">, memref<1x1x64xf16, "L1">, memref<1x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:               %[[MEM0:.*]] = memref.alloc() : memref<1x1x64xf16, "L1">
// CHECK-NEXT:               %[[MEM1:.*]] = memref.alloc() : memref<1x1x64xf16, "L1">
// CHECK-NEXT:               %[[MEM2:.*]] = memref.alloc() : memref<1x1x1x64xf16, "L1">
// CHECK-NEXT:               %[[TOK0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               %[[TOK1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:               ktdf.private_yield %[[MEM0]], %[[MEM1]], %[[MEM2]], %[[TOK0]], %[[TOK1]] : memref<1x1x64xf16, "L1">, memref<1x1x64xf16, "L1">, memref<1x1x1x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf.stage depends_in(none) depends_out(%[[PRIV]]#3) {
// CHECK-NEXT:               ktdf.data_transfer from %[[VAL_18]]{{\[}}%[[VAL_33]], %[[VAL_34]], %[[VAL_35]] * 64] size [1, 1, 64] to %[[PRIV]]#0[0, 0, 0] size [1, 1, 64] : memref<192x16x192xf16, strided<[4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<1x1x64xf16, "L1">
// CHECK-NEXT:               ktdf.data_transfer from %[[VAL_25]]{{\[}}%[[VAL_33]], %[[VAL_34]], %[[VAL_35]] * 64] size [1, 1, 64] to %[[PRIV]]#1[0, 0, 0] size [1, 1, 64] : memref<192x16x192xf16, strided<[4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<1x1x64xf16, "L1">
// CHECK-NEXT:             } {applicable_units = ["MNILU"]}
// CHECK-NEXT:             ktdf.stage depends_in(%[[PRIV]]#4) depends_out(none) {
// CHECK-NEXT:               ktdf.data_transfer from %[[PRIV]]#2[0, 0, 0, 0] size [1, 1, 1, 64] to %[[VAL_32]][0, %[[VAL_33]], %[[VAL_34]], %[[VAL_35]] * 64] size [1, 1, 1, 64] : memref<1x1x1x64xf16, "L1">, memref<1x192x16x192xf16, strided<[49152, 4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:             } {applicable_units = ["MNISU"]}
// CHECK-NEXT:           }
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:       } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }



#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 11 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  func.func private @"local-schedule-0"() attributes {grid = [4 : index, 4 : index]} {
    %c1024 = arith.constant 1024 : index
    %c786432 = arith.constant 786432 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c162304 = arith.constant 162304 : index
    %c113152 = arith.constant 113152 : index
    %c16 = arith.constant 16 : index
    %c192 = arith.constant 192 : index
    %c64000 = arith.constant 64000 : index
    %0:2 = ktdp.get_compute_tile_id : index, index
    %1 = arith.muli %0#0, %c192 : index
    %2 = ktdp.construct_memory_view %c64000, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x64x64xf16>
    %3 = arith.muli %0#0, %c786432 : index
    %4 = arith.muli %0#1, %c1024 : index
    %5 = arith.addi %3, %4 : index
    %6 = arith.addi %5, %1 : index
    %memspacecast = memref.memory_space_cast %2 : memref<12x64x64xf16> to memref<12x64x64xf16, #ktdp.memory_space<global>>
    %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [%6], sizes: [192, 16, 192], strides: [4096, 64, 1] : memref<12x64x64xf16, #ktdp.memory_space<global>> to memref<192x16x192xf16, strided<[4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>
    %7 = ktdp.construct_memory_view %c113152, sizes: [12, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x64x64xf16>
    %8 = arith.muli %0#0, %c786432 : index
    %9 = arith.muli %0#1, %c1024 : index
    %10 = arith.addi %8, %9 : index
    %11 = arith.addi %10, %1 : index
    %memspacecast_0 = memref.memory_space_cast %7 : memref<12x64x64xf16> to memref<12x64x64xf16, #ktdp.memory_space<global>>
    %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [%11], sizes: [192, 16, 192], strides: [4096, 64, 1] : memref<12x64x64xf16, #ktdp.memory_space<global>> to memref<192x16x192xf16, strided<[4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>
    %12 = ktdp.construct_memory_view %c162304, sizes: [1, 12, 64, 64], strides: [49152, 4096, 64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1x12x64x64xf16>
    %13 = arith.muli %0#0, %c786432 : index
    %14 = arith.muli %0#1, %c1024 : index
    %15 = arith.addi %13, %14 : index
    %16 = arith.addi %15, %1 : index
    %memspacecast_2 = memref.memory_space_cast %12 : memref<1x12x64x64xf16> to memref<1x12x64x64xf16, #ktdp.memory_space<global>>
    %reinterpret_cast_3 = memref.reinterpret_cast %memspacecast_2 to offset: [%16], sizes: [1, 192, 16, 192], strides: [49152, 4096, 64, 1] : memref<1x12x64x64xf16, #ktdp.memory_space<global>> to memref<1x192x16x192xf16, strided<[49152, 4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>
    scf.for %arg0 = %c0 to %c192 step %c1 {
      scf.for %arg1 = %c1 to %c16 step %c1 {
        scf.for %arg2 = %c0 to %c192 step %c64 {
          ktdf.pipeline {
            %17:10 = ktdf.private -> (memref<1x1x64xf16, "L1">, memref<1x1x64xf16, "L1">, memref<1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
              %alloc = memref.alloc() : memref<1x1x64xf16, "L1">
              %alloc_4 = memref.alloc() : memref<1x1x64xf16, "L1">
              %alloc_5 = memref.alloc() : memref<1x1x1x64xf16, "L1">
              %18:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              %19 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
              %20 = ktdf.create_token : !ktdf.token
              %21 = ktdf.create_token : !ktdf.token
              %22 = ktdf.create_token : !ktdf.token
              %23 = ktdf.create_token : !ktdf.token
              ktdf.private_yield %alloc, %alloc_4, %alloc_5, %18#0, %18#1, %19, %20, %21, %22, %23 : memref<1x1x64xf16, "L1">, memref<1x1x64xf16, "L1">, memref<1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
            }
            ktdf.stage depends_in(none) depends_out(%17#6) {
              ktdf.data_transfer from %reinterpret_cast[%arg0, %arg1 - 1, %arg2] size [1, 1, 64] to %17#0[0, 0, 0] size [1, 1, 64] : memref<192x16x192xf16, strided<[4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<1x1x64xf16, "L1">
              ktdf.data_transfer from %reinterpret_cast_1[%arg0, %arg1 - 1, %arg2] size [1, 1, 64] to %17#1[0, 0, 0] size [1, 1, 64] : memref<192x16x192xf16, strided<[4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<1x1x64xf16, "L1">
            } {applicable_units = ["MNILU"]}
            ktdf.stage depends_in(%17#9) depends_out(none) {
              ktdf.data_transfer from %17#2[0, 0, 0, 0] size [1, 1, 1, 64] to %reinterpret_cast_3[%c0, %arg0, %arg1 - 1, %arg2] size [1, 1, 1, 64] : memref<1x1x1x64xf16, "L1">, memref<1x192x16x192xf16, strided<[49152, 4096, 64, 1], offset: ?>, #ktdp.memory_space<global>>
            } {applicable_units = ["MNISU"]}
          }
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

