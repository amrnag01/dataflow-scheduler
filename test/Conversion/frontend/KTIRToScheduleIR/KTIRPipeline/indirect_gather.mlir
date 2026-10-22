// RUN: dataflow-scheduler-opt --mlir-print-local-scope --ktir-map-and-tile --ktir-bufferize --ktir-pipeline %s | FileCheck %s

// Gather: ktdp_lowering.construct_indirect_access_tile + ktdp.load becomes a
// ktdf.ind_data_transfer in the load stage, co-located with (and after) the
// guarded IAB fill, followed by the compute and the direct store stages.
//
// Input mirrors the output of IndirectAccessLoopMaterialization/basic-gather.mlir.
// The compute is already nested in loops, so ktir-map-and-tile maps it to the
// compute unit and throttles it, but does not tile it.

// CHECK-LABEL: func.func @local_schedule_1
// CHECK:         %[[AB:.+]] = ktdp.construct_memory_view {{.+}} : memref<2x32xindex, #ktdp.memory_space<global>>
// CHECK:         %[[IAB:.+]] = ktdp_lowering.construct_memory_view {{.+}} : memref<32xindex, "IAB">
// CHECK:         scf.for %[[I1:.+]] = %{{.+}} to %{{.+}} step
// CHECK:           scf.for %[[I2:.+]] = %{{.+}} to %{{.+}} step
// CHECK:             scf.for %[[I3:.+]] = %{{.+}} to %{{.+}} step
// CHECK-NEXT:          ktdf.pipeline {
// CHECK:                 %[[PRV:.+]]:4 = ktdf.private

// Stage 1 (load): IAB fill, then the indirect transfer into the load FIFO.
// CHECK:                 ktdf.stage depends_in(none) depends_out(%[[PRV]]#3) {
// CHECK-NEXT:              %[[EQ:.+]] = arith.cmpi eq, %[[I2]], %{{.+}} : index
// CHECK-NEXT:              scf.if %[[EQ]] {
// CHECK-NEXT:                %[[AB_CAST:.+]] = memref.memory_space_cast %[[AB]]
// CHECK-NEXT:                ktdf.data_transfer from %[[AB_CAST]][%[[I1]], 0] size [1, 32] to %[[IAB]][0] size [32]
// CHECK-SAME:                  {dataflow_scheduler.throttle = 1 : i64} : memref<2x32xindex, "DDR">, memref<32xindex, "IAB">
// CHECK-NEXT:              }
// CHECK:                   %[[BASE:.+]] = memref.reinterpret_cast %{{.+}} to offset: [%{{.+}}], sizes: [64, 2, 64], strides: [64, 4096, 1]
// CHECK-NEXT:              %[[IAB_RC:.+]] = memref.reinterpret_cast %[[IAB]] to offset: [%{{.+}}], sizes: [32], strides: [1]
// CHECK-NEXT:              ktdf.ind_data_transfer
// CHECK-NEXT:               ind_src = %[[IAB_RC]][%[[I2]]]
// CHECK-NEXT:               dir_src = %[[BASE]][0, %[[I3]], 0] size [1, 1, 64]
// CHECK-NEXT:               ind_dst = none
// CHECK-NEXT:               dir_dst = %[[PRV]]#2 size [64]
// CHECK-NEXT:               {dataflow_scheduler.throttle = 64 : i64} : memref<32xindex, strided<[1], offset: ?>, "IAB">, memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">, none, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
// CHECK-NEXT:            }

// Stage 2 (compute).
// CHECK-NEXT:            ktdf.stage depends_in(%[[PRV]]#3) depends_out(%[[PRV]]#1) {
// CHECK-NEXT:              %[[DATA:.+]] = ktdf.read_from_fifo %[[PRV]]#2 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK:                   %[[RES:.+]] = linalg.generic {{.+}} ins(%[[DATA]] : tensor<64xf16>) {{.+}} attrs = {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"}
// CHECK:                   ktdf.write_to_fifo %[[RES]], %[[PRV]]#0
// CHECK-NEXT:            } {applicable_units = ["SFU"]}

// Stage 3 (direct store).
// CHECK-NEXT:            ktdf.stage depends_in(%[[PRV]]#1) depends_out(none) {
// CHECK-NOT:               ktdf.ind_data_transfer
// CHECK:                   ktdf.data_transfer from %[[PRV]]#0 size [1, 1, 1, 64] to %{{.+}}[%[[I1]], %[[I2]], 0, 0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64}
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NOT:     ktdp_lowering.ind_load
// CHECK-NOT:     ktdp_lowering.construct_indirect_access_tile

#set_ab       = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_ab_pin   = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_iab      = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_desc1    = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// After loop materialization: 1 iv (%arg4, 0..63), loop %i3 is captured.
#set_vars1    = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
#set_desc2     = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set_desc2_pin = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map1  = affine_map<(d0) -> (d0)>
#map2  = affine_map<(d0, d1) -> (d0, d1)>
#map4  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @local_schedule_1 {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c0     = arith.constant 0     : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 2000 : index
    %addr_buf = ktdp.construct_memory_view %addr_buf_base,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_ab, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>

    // IAB memref: already arch-memory-space "IAB", no memory_space_cast needed.
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    %c0_i1 = arith.constant 0 : index
    %c2    = arith.constant 2 : index
    %c1    = arith.constant 1 : index
    scf.for %i1 = %c0_i1 to %c2 step %c1 {
      %c0_i2 = arith.constant 0  : index
      %c32   = arith.constant 32 : index
      scf.for %i2 = %c0_i2 to %c32 step %c1 {
        scf.for %i3 = %c0_i1 to %c2 step %c1 {
          %eq0 = arith.cmpi eq, %i2, %c0_i2 : index
          scf.if %eq0 {
            %addr_at = ktdp.construct_access_tile %addr_buf[%i1, %c0]
                {access_tile_order = #map2, access_tile_set = #set_ab_pin}
                : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x32xindex>
            %addr_stick_2d = ktdp.load %addr_at : !ktdp.access_tile<1x32xindex> -> tensor<1x32xindex>
            %addr_stick = tensor.collapse_shape %addr_stick_2d [[0, 1]]
                : tensor<1x32xindex> into tensor<32xindex>
            %iab_at = ktdp.construct_access_tile %iab_mv[%c0]
                {access_tile_order = #map1, access_tile_set = #set_iab}
                : memref<32xindex, "IAB"> -> !ktdp.access_tile<32xindex>
            ktdp.store %addr_stick, %iab_at : tensor<32xindex>, !ktdp.access_tile<32xindex>
          }

          // After loop materialization: %i3 is captured, %arg4 is the sole iv.
          // captured = [%c0, %i3], intermediate = [%arg4]
          %desc_1 = ktdp.construct_memory_view %c0,
              sizes: [64, 2, 64], strides: [64, 4096, 1]
              {coordinate_set = #set_desc1, memory_space = #ktdp.memory_space<global>}
              : memref<64x2x64xf16>

          %tmp1 = ktdp_lowering.construct_indirect_access_tile
              intermediate_variables(%arg4)
              base_ptr = %iab_mv[%i2]
              %desc_1[%c0, (%c0 + %i3), %arg4]
              {variables_space_order = #map1, variables_space_set = #set_vars1}
              : memref<64x2x64xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<64xindex>
          %tmp1_data = ktdp.load %tmp1 : !ktdp.access_tile<64xindex> -> tensor<64xf16>

          %e = tensor.empty() : tensor<64xf16>
          %computed = linalg.generic {
              indexing_maps = [#map1, #map1],
              iterator_types = ["parallel"]}
              ins(%tmp1_data : tensor<64xf16>)
              outs(%e : tensor<64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %cf10 = arith.constant 10.000000e+00 : f16
            %added = arith.addf %in, %cf10 : f16
            linalg.yield %added : f16
          } -> tensor<64xf16>

          %expanded = tensor.expand_shape %computed [[0, 1, 2, 3]] output_shape [1, 1, 1, 64]
              : tensor<64xf16> into tensor<1x1x1x64xf16>
          %desc_2 = ktdp.construct_memory_view %c10000,
              sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
              {coordinate_set = #set_desc2, memory_space = #ktdp.memory_space<global>}
              : memref<2x32x2x64xf16>
          %desc_2_at = ktdp.construct_access_tile %desc_2[%i1, %i2, %c0, %c0]
              {access_tile_order = #map4, access_tile_set = #set_desc2_pin}
              : memref<2x32x2x64xf16> -> !ktdp.access_tile<1x1x1x64xindex>
          ktdp.store %expanded, %desc_2_at : tensor<1x1x1x64xf16>, !ktdp.access_tile<1x1x1x64xindex>
        }
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
