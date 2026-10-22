// RUN: dataflow-scheduler-opt --indirect-addr-buf-fill-legalization %s | FileCheck %s

// As scatter.mlir, but the fill's source view is built inside the fill stage,
// under the fill's guard. The staging stage must recreate that view chain
// rather than reference values it cannot see, and the fill stage must no
// longer keep it around.

// CHECK-LABEL: func.func @scatter_source_in_stage() {
// CHECK:         %[[PRV:.+]]:9 = ktdf.private

// The load unit rebuilds the source view under the guard and stages from it.
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[PRV]]#4) {
// CHECK-NEXT:      ktdf.data_transfer from %{{.*}}[0, 0] size [2, 64] to %[[PRV]]#0[0, 0] size [2, 64]
// CHECK-NEXT:      %[[EQ:.+]] = arith.cmpi eq, %[[IV:.+]], %[[C0:.+]] : index
// CHECK-NEXT:      scf.if %[[EQ]] {
// CHECK-NEXT:        %[[OFF:.+]] = arith.muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        %[[SRC:.+]] = memref.reinterpret_cast %{{.*}} to offset: [%[[OFF]]], sizes: [32], strides: [1]
// CHECK-NEXT:        ktdf.data_transfer from %[[SRC]]{{\[}}%[[C0]]] size [32] to %[[PRV]]#8[0] size [32] : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, "L1">
// CHECK-NEXT:      }
// CHECK-NEXT:    } {applicable_units = ["MNILU"]}

// The fill stage keeps only the rewritten fill; the old view chain is gone.
// CHECK:         ktdf.stage depends_in(%[[PRV]]#7) depends_out(none) {
// CHECK-NEXT:      %[[EQ_ST:.+]] = arith.cmpi eq, %[[IV]], %[[C0]] : index
// CHECK-NEXT:      scf.if %[[EQ_ST]] {
// CHECK-NEXT:        ktdf.data_transfer from %[[PRV]]#8[0] size [32] to %{{.*}}{{\[}}%[[C0]]] size [32] : memref<32xindex, "L1">, memref<32xindex, strided<[1], offset: ?>, "IAB">
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.ind_data_transfer

#map = affine_map<(d0, d1) -> (d0, d1)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set3 = affine_set<(d2, d3) : (d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @scatter_source_in_stage() {
    %c0     = arith.constant 0 : index
    %c1     = arith.constant 1 : index
    %c2     = arith.constant 2 : index
    %c32    = arith.constant 32 : index
    %c128   = arith.constant 128 : index
    %c4096  = arith.constant 4096 : index
    %c1000  = arith.constant 1000 : index
    %c10000 = arith.constant 10000 : index

    %desc_src_mv = ktdp.construct_memory_view %c1000,
        sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>
    %desc_src_msc = memref.memory_space_cast %desc_src_mv
        : memref<2x32x2x64xf16> to memref<2x32x2x64xf16, "DDR">

    %desc_dst_mv = ktdp.construct_memory_view %c10000,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>
    %desc_dst_msc = memref.memory_space_cast %desc_dst_mv
        : memref<64x2x64xf16> to memref<64x2x64xf16, "DDR">
    %desc_dst_rc = memref.reinterpret_cast %desc_dst_msc
        to offset: [0], sizes: [64, 2, 64], strides: [64, 4096, 1]
        : memref<64x2x64xf16, "DDR">
          to memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">

    %addr_buf_base = arith.constant 999 : index
    %addr_buf_mv = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set3, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    %addr_buf_msc = memref.memory_space_cast %addr_buf_mv
        : memref<2x32xindex, #ktdp.memory_space<global>>
          to memref<2x32xindex, "DDR">

    scf.for %i1 = %c0 to %c2 step %c1 {
      scf.for %i2 = %c0 to %c32 step %c1 {
        %off_i1 = arith.muli %i1, %c4096 : index
        %off_i2 = arith.muli %i2, %c128 : index
        %src_off = arith.addi %off_i1, %off_i2 : index
        %desc_src_rc = memref.reinterpret_cast %desc_src_msc
            to offset: [%src_off], sizes: [2, 64], strides: [64, 1]
            : memref<2x32x2x64xf16, "DDR">
              to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">

        %iab = memref.alloc() : memref<32xindex, "IAB">
        %iab_rc = memref.reinterpret_cast %iab
            to offset: [0], sizes: [32], strides: [1]
            : memref<32xindex, "IAB">
              to memref<32xindex, strided<[1], offset: ?>, "IAB">

        ktdf.pipeline {
          %prv:8 = ktdf.private -> (
              memref<2x64xf16, "L1">,
              !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>,
              !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>,
              memref<2x64xf16, "L1">,
              !ktdf.token,
              !ktdf.token,
              !ktdf.token,
              !ktdf.token
          ) {
            %l1_ld = memref.alloc() : memref<2x64xf16, "L1">
            %fifo_l1_sfu = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
            %fifo_sfu_l1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>
            %l1_st = memref.alloc() : memref<2x64xf16, "L1">
            %tok0 = ktdf.create_token : !ktdf.token
            %tok1 = ktdf.create_token : !ktdf.token
            %tok2 = ktdf.create_token : !ktdf.token
            %tok3 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %l1_ld, %fifo_l1_sfu, %fifo_sfu_l1, %l1_st, %tok0, %tok1, %tok2, %tok3
                : memref<2x64xf16, "L1">,
                  !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>,
                  !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>,
                  memref<2x64xf16, "L1">,
                  !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
          }

          // Stage 1 (MNILU): DDR -> L1
          ktdf.stage depends_in(none) depends_out(%prv#4) {
            ktdf.data_transfer
                from %desc_src_rc[0, 0] size [2, 64]
                to   %prv#0[0, 0]       size [2, 64]
                : memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">,
                  memref<2x64xf16, "L1">
          } {applicable_units = ["MNILU"]}

          // Stage 2 (L1LU): L1 -> SFU FIFO
          ktdf.stage depends_in(%prv#4) depends_out(%prv#5) {
            ktdf.data_transfer
                from %prv#0[0, 0] size [2, 64]
                to   %prv#1       size [128]
                : memref<2x64xf16, "L1">,
                  !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
          } {applicable_units = ["L1LU"]}

          // Stage 3 (SFU): compute
          ktdf.stage depends_in(%prv#5) depends_out(%prv#6) {
            %src = ktdf.read_from_fifo %prv#1
                : !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16> -> tensor<2x64xf16>
            %empty = tensor.empty() : tensor<2x64xf16>
            %result = linalg.generic {
                indexing_maps = [#map, #map],
                iterator_types = ["parallel", "parallel"]
            } ins(%src : tensor<2x64xf16>) outs(%empty : tensor<2x64xf16>) {
            ^bb0(%in: f16, %out: f16):
              %cf10 = arith.constant 10.000000e+00 : f16
              %sum = arith.addf %in, %cf10 : f16
              linalg.yield %sum : f16
            } -> tensor<2x64xf16>
            ktdf.write_to_fifo %result, %prv#2
                : tensor<2x64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>
          } {applicable_units = ["SFU"]}

          // Stage 4 (L1SU): SFU FIFO -> L1
          ktdf.stage depends_in(%prv#6) depends_out(%prv#7) {
            ktdf.data_transfer
                from %prv#2       size [128]
                to   %prv#3[0, 0] size [2, 64]
                : !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>,
                  memref<2x64xf16, "L1">
          } {applicable_units = ["L1SU"]}

          // Stage 5 (MNISU): illegal DDR -> IAB fill whose source view is built
          // under the guard, plus the L1 -> DDR scatter
          ktdf.stage depends_in(%prv#7) depends_out(none) {
            %eq0 = arith.cmpi eq, %i2, %c0 : index
            scf.if %eq0 {
              %addr_off = arith.muli %i1, %c32 : index
              %addr_buf_rc = memref.reinterpret_cast %addr_buf_msc
                  to offset: [%addr_off], sizes: [32], strides: [1]
                  : memref<2x32xindex, "DDR">
                    to memref<32xindex, strided<[1], offset: ?>, "DDR">
              ktdf.data_transfer
                  from %addr_buf_rc[%c0] size [32]
                  to   %iab_rc[%c0]      size [32]
                  : memref<32xindex, strided<[1], offset: ?>, "DDR">,
                    memref<32xindex, strided<[1], offset: ?>, "IAB">
            }
            ktdf.ind_data_transfer
                ind_src = none
                dir_src = %prv#3[%c0, %c0]            size [2, 64]
                ind_dst = %iab_rc[%i2]
                dir_dst = %desc_dst_rc[%c0, %c0, %c0] size [1, 2, 64]
                : none,
                  memref<2x64xf16, "L1">,
                  memref<32xindex, strided<[1], offset: ?>, "IAB">,
                  memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">
          } {applicable_units = ["MNISU"]}
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
