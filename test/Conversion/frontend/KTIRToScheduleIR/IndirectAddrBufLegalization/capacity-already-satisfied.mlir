// RUN: dataflow-scheduler-opt --mlir-print-local-scope --indirect-addr-buf-legalization %s | FileCheck %s

// IAB capacity already satisfied: the ind_addr_buf view already has exactly
// one row (32 entries == the 32-entry hardware capacity). The window loop
// materialization should recognise that no window loop is needed (W = 0) and
// the entry loop materialization should emit only the inner per-entry loop,
// directly from function scope.

// CHECK-LABEL: func.func @capacity_satisfied
// No window loop needed for W = 0 (single 32-entry row); the IAB view is a
// pure descriptor for a fixed hardware region, so it is hoisted directly to
// function scope, right before the entry loop — no iter-arg needed.
// CHECK-NOT:  loop_type
// CHECK:      %[[IAB_VIEW:.+]] = ktdp_lowering.construct_memory_view %c0, sizes: [32],
// kept: the IAB is already exactly 32 entries (W=0), so coordinate_set is unchanged.
// CHECK-SAME:   coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK-SAME:   memory_space = "IAB"
// CHECK:      scf.for %[[I2:.+]] = %c0{{.*}} to %c32 step %c1{{.*}} {
// CHECK:        %[[EQ0:.+]] = arith.cmpi eq, %[[I2]], %c0
// CHECK:        scf.if %[[EQ0]] {
// CHECK:          ktdp.construct_access_tile {{.*}}[%c0]
// kept: no window dim exists (W=0), so this tile spans the full 0..31 range untouched.
// CHECK-SAME:       access_tile_order = affine_map<(d0) -> (d0)>
// CHECK-SAME:       access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK-SAME:       -> !ktdp.access_tile<32xindex>
// CHECK:          ktdp.load {{.*}} <32xindex> -> tensor<32xindex>
// CHECK:          ktdp.construct_access_tile %[[IAB_VIEW]][%c0]
// kept: same access_tile_set/order as the IAB view's own coordinate_set.
// CHECK-SAME:       access_tile_order = affine_map<(d0) -> (d0)>
// CHECK-SAME:       access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK:          ktdp.store {{.*}} tensor<32xindex>, <32xindex>
// CHECK:        }
// CHECK:        ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME:     base_ptr = %[[IAB_VIEW]][%[[I2]]]
// dropped: entry dim vanishes (absorbed by base_ptr[iv]); the two direct dims kept unchanged.
// CHECK-SAME:     variables_space_order = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-SAME:     variables_space_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-SAME:     -> !ktdp.access_tile<2x64xindex>
// CHECK:        ktdp.load {{.*}} <2x64xindex> -> tensor<2x64xf16>
// CHECK:        linalg.generic
// CHECK-SAME:     iterator_types = ["parallel", "parallel"]
// CHECK:        tensor.expand_shape {{.*}} tensor<2x64xf16> into tensor<1x2x64xf16>
// CHECK:        ktdp.construct_access_tile {{.*}}{{\[}}%[[I2]], {{.*}}{{\]}}
// pinned: entry dim (%[[I2]]) collapses to a single point; the two direct dims stay full-range.
// CHECK-SAME:     access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-SAME:     access_tile_set = affine_set<(d0, d1, d2) : (d0 == 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK-SAME:     -> !ktdp.access_tile<1x2x64xindex>
// CHECK:        ktdp.store {{.*}} tensor<1x2x64xf16>, <1x2x64xindex>
// CHECK:      }

#set_iab_1x32 = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_vars     = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#map_vars     = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1         = affine_map<(d0) -> (d0)>

module @local_schedule_1 {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @capacity_satisfied(%arg6: index, %arg7: index, %arg8: index)
      attributes {grid = [1 : index]} {
    %c0     = arith.constant 0     : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 3000 : index

    // Single 32-entry row: already within the hardware capacity, so
    // the window loop is a no-op (W = 0).
    %addr_buf = ktdp.construct_memory_view %addr_buf_base,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab_1x32, memory_space = #ktdp.memory_space<global>}
        : memref<32xindex, #ktdp.memory_space<global>>
    %addr_buf_at = ktdp.construct_access_tile %addr_buf[%c0]
        {access_tile_order = #map1, access_tile_set = #set_iab_1x32}
        : memref<32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<32xindex>
    %addr_tensor = ktdp.load %addr_buf_at
        : !ktdp.access_tile<32xindex> -> tensor<32xindex>

    // 1-D IAB: already within the 32-entry hardware limit
    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab_1x32, memory_space = "IAB"}
        : memref<32xindex, "IAB">
    %iab_at = ktdp.construct_access_tile %iab_mv[%c0]
        {access_tile_order = #map1, access_tile_set = #set_iab_1x32}
        : memref<32xindex, "IAB"> -> !ktdp.access_tile<32xindex>
    ktdp.store %addr_tensor, %iab_at
        : tensor<32xindex>, !ktdp.access_tile<32xindex>

    %desc_1 = ktdp.construct_memory_view %c0,
        sizes: [64, 64], strides: [64, 1]
        {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
         memory_space = #ktdp.memory_space<global>}
        : memref<64x64xf16>

    // Result shape 32×2×64 reflects the full IAB window (32 entries) ×
    // the two direct dims (arg7 spans a 2-element slice, arg8 the full 64).
    %tile = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg6]
        %desc_1[(%arg7), (%arg8)]
        {variables_space_order = #map_vars,
         variables_space_set = #set_vars}
        : memref<64x64xf16>, memref<32xindex, "IAB">
        -> !ktdp.access_tile<32x2x64xindex>
    %tile_0 = ktdp.load %tile
        : !ktdp.access_tile<32x2x64xindex> -> tensor<32x2x64xf16>

    // Element-wise compute on the full 32×2×64 tile (3 parallel dims).
    %empty = tensor.empty() : tensor<32x2x64xf16>
    %computed = linalg.generic {
        indexing_maps = [#map_vars, #map_vars],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%tile_0 : tensor<32x2x64xf16>)
        outs(%empty : tensor<32x2x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %cf10 = arith.constant 10.000000e+00 : f16
      %added = arith.addf %in, %cf10 : f16
      linalg.yield %added : f16
    } -> tensor<32x2x64xf16>

    %desc_2 = ktdp.construct_memory_view %c10000,
        sizes: [32, 2, 64], strides: [128, 64, 1]
        {coordinate_set = #set_vars, memory_space = #ktdp.memory_space<global>}
        : memref<32x2x64xf16>
    %desc_2_at = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0]
        {access_tile_order = #map_vars, access_tile_set = #set_vars}
        : memref<32x2x64xf16> -> !ktdp.access_tile<32x2x64xindex>
    ktdp.store %computed, %desc_2_at
        : tensor<32x2x64xf16>, !ktdp.access_tile<32x2x64xindex>
    return
  }
}
