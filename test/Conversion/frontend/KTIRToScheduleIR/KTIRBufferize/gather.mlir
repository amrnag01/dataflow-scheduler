// RUN: dataflow-scheduler-opt --mlir-print-local-scope --ktir-bufferize %s | FileCheck %s

// Gather (indirect load): ktdp_lowering.construct_indirect_access_tile +
// ktdp.load  ->  ktdp_lowering.ind_load
//
// Input mirrors the output of IndirectAccessLoopMaterialization/basic-gather.mlir:
//   - outer loops: scf.for %i1 (0..2), scf.for %i2 (0..32), scf.for %i3 (0..2)
//   - inside %i3: scf.if guard for IAB fill; then the indirect access.
//   - the indirect tile has 1 iv (%arg4, 64 elements, sole remaining variable).
//   - the tile is !ktdp.access_tile<64xindex> (rank 1).
//
// Expected output:
//   - The IAB memref is reinterpret_cast-only (no memory_space_cast).
//   - The base memref is memory_space_cast to "DDR" then full-rank
//     reinterpret_cast with offset 0 and the construct_memory_view strides.
//   - ktdp_lowering.ind_load with IAB index = %i2, offsets [0, %i3, 0],
//     sizes [1, 1, 64], strides [1, 1, 1].
//   - The IAB fill inside the scf.if becomes ktdf.data_transfer.
//   - No ktdp_lowering.construct_indirect_access_tile remains.

// CHECK-LABEL: func.func @local_schedule_1
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.if
// CHECK:               ktdf.data_transfer
// CHECK:             memref.memory_space_cast {{%.+}} to memref<64x2x64xf16, "DDR">
// CHECK:             memref.reinterpret_cast {{%.+}} to offset: {{.+}} sizes: [64, 2, 64], strides: [64, 4096, 1]
// CHECK-SAME:            "DDR">
// CHECK:             memref.reinterpret_cast {{%.+}} to offset: {{.+}} sizes: [32], strides: [1]
// CHECK-SAME:            "IAB">
// CHECK:             ktdp_lowering.ind_load {{%.+}} [1, 1, 64] [1, 1, 1]
// CHECK-SAME:            strided<[1], offset: ?>
// CHECK-SAME:            strided<[64, 4096, 1], offset: ?>
// CHECK-SAME:            -> tensor<64xf16>
// CHECK:             linalg.generic
// CHECK:             ktdp_lowering.store

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
