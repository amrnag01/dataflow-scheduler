// RUN: dataflow-scheduler-opt --mlir-print-local-scope --ktir-bufferize %s | FileCheck %s

// Scatter (indirect store): ktdp_lowering.construct_indirect_access_tile +
// ktdp.store  ->  ktdp_lowering.ind_store
//
// Input mirrors the output of IndirectAccessLoopMaterialization/basic-scatter.mlir
// after loop materialization: %i3 (0..2) is the outermost materialized loop,
// %i2 (0..32) is the entry loop. The indirect tile has 1 iv (%arg4, 64 elems).
//
// Expected output:
//   - The IAB memref is reinterpret_cast-only (no memory_space_cast).
//   - The base (dest) memref is memory_space_cast + full-rank reinterpret_cast.
//   - ktdp_lowering.ind_store with IAB index = %i2, offsets [0, %i3, 0],
//     sizes [1, 1, 64], strides [1, 1, 1].
//   - The IAB fill inside the scf.if becomes ktdf.data_transfer.
//   - The source direct-load also becomes a ktdf.data_transfer.

// CHECK-LABEL: func.func @local_schedule_1
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.if
// CHECK:               ktdf.data_transfer
// CHECK:             ktdp_lowering.load
// CHECK:             memref.memory_space_cast {{%.+}} to memref<64x2x64xf16, "DDR">
// CHECK:             memref.reinterpret_cast {{%.+}} to offset: {{.+}} sizes: [64, 2, 64], strides: [64, 4096, 1]
// CHECK-SAME:            "DDR">
// CHECK:             memref.reinterpret_cast {{%.+}} to offset: {{.+}} sizes: [32], strides: [1]
// CHECK-SAME:            "IAB">
// CHECK:             ktdp_lowering.ind_store {{%.+}} [1, 1, 64] [1, 1, 1]
// CHECK-SAME:            tensor<64xf16>
// CHECK-SAME:            strided<[1], offset: ?>
// CHECK-SAME:            strided<[64, 4096, 1], offset: ?>

#set_ab       = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_ab_pin   = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_iab      = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_desc_dst = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// After loop materialization: 1 iv (%arg4, 0..63), loop %i3 is captured.
#set_vars1    = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
#set_src      = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set_src_pin  = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @local_schedule_1 {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c0    = arith.constant 0    : index
    %c1000 = arith.constant 1000 : index
    %addr_buf_base = arith.constant 2000 : index
    %addr_buf = ktdp.construct_memory_view %addr_buf_base,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_ab, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    // Source: 4-D memref, access via direct tile (collapsed to 2D before compute).
    %desc_src = ktdp.construct_memory_view %c1000,
        sizes: [2, 32, 1, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set_src, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x1x64xf16>

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

          // Source-tile load: direct access, no indirect dim.
          %src_at = ktdp.construct_access_tile %desc_src[%i1, %i2, %c0, %c0]
              {access_tile_order = #map4, access_tile_set = #set_src_pin}
              : memref<2x32x1x64xf16> -> !ktdp.access_tile<1x1x1x64xindex>
          %src_load = ktdp.load %src_at : !ktdp.access_tile<1x1x1x64xindex> -> tensor<1x1x1x64xf16>
          %src = tensor.collapse_shape %src_load [[0, 1, 2, 3]]
              : tensor<1x1x1x64xf16> into tensor<64xf16>

          %e = tensor.empty() : tensor<64xf16>
          %computed = linalg.generic {
              indexing_maps = [#map1, #map1],
              iterator_types = ["parallel"]}
              ins(%src : tensor<64xf16>)
              outs(%e : tensor<64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %cf10 = arith.constant 10.000000e+00 : f16
            %added = arith.addf %in, %cf10 : f16
            linalg.yield %added : f16
          } -> tensor<64xf16>

          // Indirect destination: 1 iv (%arg4), captured = [%c0, %i3].
          %desc_dst = ktdp.construct_memory_view %c0,
              sizes: [64, 2, 64], strides: [64, 4096, 1]
              {coordinate_set = #set_desc_dst, memory_space = #ktdp.memory_space<global>}
              : memref<64x2x64xf16>

          %dst_tile = ktdp_lowering.construct_indirect_access_tile
              intermediate_variables(%arg4)
              base_ptr = %iab_mv[%i2]
              %desc_dst[%c0, (%c0 + %i3), %arg4]
              {variables_space_order = #map1, variables_space_set = #set_vars1}
              : memref<64x2x64xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<64xindex>
          ktdp.store %computed, %dst_tile : tensor<64xf16>, !ktdp.access_tile<64xindex>
        }
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
