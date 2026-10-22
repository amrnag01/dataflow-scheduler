// RUN: dataflow-scheduler-opt --mlir-print-local-scope --ktir-bufferize %s | FileCheck %s

// Gather after map-and-tile: ktdp_lowering.construct_indirect_access_tile +
// ktdp_lowering.load (on the access tile, with a non-full slice)
// ->  ktdp_lowering.ind_load
//
// This tests the ktdp_lowering::LoadOp path in LowerIndirectLoad, where the
// tile was already sliced by ktir-map-and-tile.  In this scenario the access
// tile rank is 1 (64 elements, one iv), and the slice takes the first 32.
//
// Expected output:
//   - ind_load with sizes [1, 1, 32] (the slice size) rather than [1, 1, 64].
//   - The offset for the direct dim 1 is captured variable %i3 (0 here).
//   - IAB and base memrefs are handled as usual.

// CHECK-LABEL: func.func @local_schedule_after_mat
// CHECK:       memref.memory_space_cast {{%.+}} to memref<64x2x64xf16, "DDR">
// CHECK:       memref.reinterpret_cast {{%.+}} to offset: {{.+}} sizes: [64, 2, 64], strides: [64, 4096, 1]
// CHECK-SAME:      "DDR">
// CHECK:       memref.reinterpret_cast {{%.+}} to offset: {{.+}} sizes: [32], strides: [1]
// CHECK-SAME:      "IAB">
// CHECK:       ktdp_lowering.ind_load {{%.+}} [1, 1, 32] [1, 1, 1]
// CHECK-SAME:      strided<[1], offset: ?>
// CHECK-SAME:      strided<[64, 4096, 1], offset: ?>
// CHECK-SAME:      -> tensor<32xf16>

#set_iab   = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_desc1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// 1 iv (%arg0), 0..63
#set_vars1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

module @local_schedule_after_mat {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_after_mat() attributes {grid = [1 : index]} {
    %c0  = arith.constant 0  : index
    %iab_idx = arith.constant 5 : index

    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    %desc_1 = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set_desc1, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>

    // After map-and-tile: captured = [%c0, %c0 (constant for i3)], intermediate = [%arg0].
    // per-dim maps: [dim0 -> 0, dim1 -> captured[1] = %c0, dim2 -> %arg0].
    %tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg0)
        base_ptr = %iab_mv[%iab_idx]
        %desc_1[%c0, %c0, %arg0]
        {variables_space_order = #map1, variables_space_set = #set_vars1}
        : memref<64x2x64xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<64xindex>

    // Simulate ktir-map-and-tile slicing the tile: take first 32 of 64.
    %loaded = ktdp_lowering.load %tile [0] [32] [1]
        : !ktdp.access_tile<64xindex> -> tensor<32xf16>

    // Dummy use.
    %e = tensor.empty() : tensor<32xf16>
    %result = linalg.generic {
        indexing_maps = [#map1, #map1],
        iterator_types = ["parallel"]}
        ins(%loaded : tensor<32xf16>)
        outs(%e : tensor<32xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cf2 = arith.constant 2.0 : f16
      %scaled = arith.mulf %in, %cf2 : f16
      linalg.yield %scaled : f16
    } -> tensor<32xf16>

    // Output store to a 1D memref (direct, 32 elements).
    %out_mv = ktdp.construct_memory_view %c0,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = #ktdp.memory_space<global>}
        : memref<32xf16>
    %out_at = ktdp.construct_access_tile %out_mv[%c0]
        {access_tile_set = #set_iab, access_tile_order = #map1}
        : memref<32xf16> -> !ktdp.access_tile<32xindex>
    ktdp.store %result, %out_at : tensor<32xf16>, !ktdp.access_tile<32xindex>
    return
  }
}
