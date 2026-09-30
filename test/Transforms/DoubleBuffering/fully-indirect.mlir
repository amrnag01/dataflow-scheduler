// RUN: dataflow-scheduler-opt %s -double-buffering | FileCheck %s

// CHECK-LABEL:   func.func @fully_indirect(
// CHECK-SAME:        %[[SRC:[a-zA-Z0-9_]+]]: memref<64xf16, #ktdp.memory_space<global>>, %[[DST:[a-zA-Z0-9_]+]]: memref<64xf16, #ktdp.memory_space<global>>, %[[IAB_SRC:[a-zA-Z0-9_]+]]: memref<32xindex, #ktdp.memory_space<ct_local>>, %[[IAB_DST:[a-zA-Z0-9_]+]]: memref<32xindex, #ktdp.memory_space<ct_local>>) {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT:      %[[X0:.+]] = memref.alloc() : memref<64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:      %[[X1:.+]] = memref.alloc() : memref<64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:      scf.for %[[I:.+]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK-NEXT:        %[[PHASE:.+]] = ktdf.buffer_phase(%[[I]]) {num_phases = 2 : i64} : index
// CHECK-NEXT:        %[[SEL:.+]] = ktdf.select_memref %[[PHASE]][%[[X0]], %[[X1]]] : memref<64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:        ktdf.pipeline modulo(size : 2) {
// CHECK-NEXT:          %[[TKS:.+]]:2 = ktdf.private -> (!ktdf.token, !ktdf.token) {
// CHECK-NEXT:            %[[T1:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            %[[T2:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            ktdf.private_yield %[[T1]], %[[T2]] : !ktdf.token, !ktdf.token
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(none) depends_out(%[[TKS]]#0) {
// CHECK-NEXT:            ktdf.ind_data_transfer
// CHECK-NEXT:                ind_src = %[[IAB_SRC]][%[[I]]]
// CHECK-NEXT:                dir_src = %[[SRC]][%[[C0]]] size [64]
// CHECK-NEXT:                ind_dst = none
// CHECK-NEXT:                dir_dst = %[[SEL]][%[[C0]]] size [64]
// CHECK-NEXT:                : memref<32xindex, #ktdp.memory_space<ct_local>>, memref<64xf16, #ktdp.memory_space<global>>, none, memref<64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(%[[TKS]]#0) depends_out(%[[TKS]]#1) {
// CHECK-NEXT:            ktdf.ind_data_transfer
// CHECK-NEXT:                ind_src = none
// CHECK-NEXT:                dir_src = %[[SEL]][%[[C0]]] size [64]
// CHECK-NEXT:                ind_dst = %[[IAB_DST]][%[[I]]]
// CHECK-NEXT:                dir_dst = %[[DST]][%[[C0]]] size [64]
// CHECK-NEXT:                : none, memref<64xf16, #ktdp.memory_space<ct_local>>, memref<32xindex, #ktdp.memory_space<ct_local>>, memref<64xf16, #ktdp.memory_space<global>>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.dealloc %[[X0]] : memref<64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:      memref.dealloc %[[X1]] : memref<64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:      return
// CHECK-NEXT:    }

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")
func.func @fully_indirect(%src: memref<64xf16, #ktdp.memory_space<global>>,
                          %dst: memref<64xf16, #ktdp.memory_space<global>>,
                          %iab_src: memref<32xindex, #ktdp.memory_space<ct_local>>,
                          %iab_dst: memref<32xindex, #ktdp.memory_space<ct_local>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %l1, %t1, %t2 = ktdf.private -> (
          memref<64xf16, #ktdp.memory_space<ct_local>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdp.memory_space<ct_local>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1) {
        ktdf.ind_data_transfer
            ind_src = %iab_src[%i]
            dir_src = %src[%c0] size [64]
            ind_dst = none
            dir_dst = %l1[%c0] size [64]
            : memref<32xindex, #ktdp.memory_space<ct_local>>,
              memref<64xf16, #ktdp.memory_space<global>>,
              none,
              memref<64xf16, #ktdp.memory_space<ct_local>>
      }
      ktdf.stage depends_in(%t1) depends_out(%t2) {
        ktdf.ind_data_transfer
            ind_src = none
            dir_src = %l1[%c0] size [64]
            ind_dst = %iab_dst[%i]
            dir_dst = %dst[%c0] size [64]
            : none,
              memref<64xf16, #ktdp.memory_space<ct_local>>,
              memref<32xindex, #ktdp.memory_space<ct_local>>,
              memref<64xf16, #ktdp.memory_space<global>>
      }
    }
  }
  return
}
