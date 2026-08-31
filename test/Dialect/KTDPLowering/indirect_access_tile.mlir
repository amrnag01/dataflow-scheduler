// RUN: dataflow-scheduler-opt %s | dataflow-scheduler-opt | FileCheck %s

// Verifies round-trip parsing and printing of
// ktdp_lowering.construct_indirect_access_tile.

// CHECK-DAG: #[[MAP:.*]]  = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[SET1:.*]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK-DAG: #[[SET2:.*]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>

#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#map  = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @roundtrip_2d_iab(
// CHECK-SAME: [[BASE:%arg[0-9]+]]: memref<64x2x64xf16>
// CHECK-SAME: [[IAB:%arg[0-9]+]]: memref<2x32xindex, "IAB">
// CHECK-SAME: [[C0:%arg[0-9]+]]: index
// CHECK-SAME: [[S0:%arg[0-9]+]]: index
// CHECK-SAME: [[S1:%arg[0-9]+]]: index
// CHECK-SAME: [[D0:%arg[0-9]+]]: index
// CHECK-SAME: [[D1:%arg[0-9]+]]: index
// CHECK: ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME: intermediate_variables([[S0]], [[S1]], [[D0]], [[D1]])
// CHECK-SAME: base_ptr = [[IAB]]{{\[}}[[S0]], [[S1]]{{\]}}
// CHECK-SAME: [[BASE]]{{\[}}[[C0]], [[D0]], [[D1]]{{\]}}
// CHECK-SAME: variables_space_order = #[[MAP]]
// CHECK-SAME: variables_space_set = #[[SET1]]
// CHECK-SAME: : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
// CHECK-SAME: -> !ktdp.access_tile<2x32x2x64xindex>
func.func @roundtrip_2d_iab(
    %base : memref<64x2x64xf16>,
    %iab  : memref<2x32xindex, "IAB">,
    %c0   : index,
    %arg5 : index,
    %arg6 : index,
    %arg7 : index,
    %arg8 : index) {
  %tile = ktdp_lowering.construct_indirect_access_tile
      intermediate_variables(%arg5, %arg6, %arg7, %arg8)
      base_ptr = %iab[%arg5, %arg6]
      %base[%c0, %arg7, %arg8]
      {variables_space_set = #set1, variables_space_order = #map}
      : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
      -> !ktdp.access_tile<2x32x2x64xindex>
  return
}

// CHECK-LABEL: func @roundtrip_1d_iab(
// CHECK-SAME: [[BASE:%arg[0-9]+]]: memref<64x2x64xf16>
// CHECK-SAME: [[IAB:%arg[0-9]+]]: memref<32xindex, "IAB">
// CHECK-SAME: [[C0:%arg[0-9]+]]: index
// CHECK-SAME: [[S0:%arg[0-9]+]]: index
// CHECK-SAME: [[D0:%arg[0-9]+]]: index
// CHECK-SAME: [[D1:%arg[0-9]+]]: index
// CHECK: ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME: intermediate_variables([[S0]], [[D0]], [[D1]])
// CHECK-SAME: base_ptr = [[IAB]]{{\[}}[[S0]]{{\]}}
// CHECK-SAME: [[BASE]]{{\[}}[[C0]], [[D0]], [[D1]]{{\]}}
// CHECK-SAME: variables_space_order = #[[MAP]]
// CHECK-SAME: variables_space_set = #[[SET2]]
// CHECK-SAME: : memref<64x2x64xf16>, memref<32xindex, "IAB">
// CHECK-SAME: -> !ktdp.access_tile<32x2x64xindex>
func.func @roundtrip_1d_iab(
    %base : memref<64x2x64xf16>,
    %iab  : memref<32xindex, "IAB">,
    %c0   : index,
    %arg6 : index,
    %arg7 : index,
    %arg8 : index) {
  %tile = ktdp_lowering.construct_indirect_access_tile
      intermediate_variables(%arg6, %arg7, %arg8)
      base_ptr = %iab[%arg6]
      %base[%c0, %arg7, %arg8]
      {variables_space_set = #set2, variables_space_order = #map}
      : memref<64x2x64xf16>, memref<32xindex, "IAB">
      -> !ktdp.access_tile<32x2x64xindex>
  return
}

// CHECK-LABEL: func @mixed_iab_subscripts(
// CHECK-SAME: [[BASE:%arg[0-9]+]]: memref<64x2x64xf16>
// CHECK-SAME: [[IAB:%arg[0-9]+]]: memref<2x32xindex, "IAB">
// CHECK-SAME: [[C0:%arg[0-9]+]]: index
// CHECK-SAME: [[I1:%arg[0-9]+]]: index
// CHECK-SAME: [[S0:%arg[0-9]+]]: index
// CHECK-SAME: [[D0:%arg[0-9]+]]: index
// CHECK-SAME: [[D1:%arg[0-9]+]]: index
// CHECK: ktdp_lowering.construct_indirect_access_tile
// CHECK-SAME: intermediate_variables([[S0]], [[D0]], [[D1]])
// CHECK-SAME: base_ptr = [[IAB]]{{\[}}[[I1]], [[S0]]{{\]}}
// CHECK-SAME: [[BASE]]{{\[}}[[C0]], [[D0]], [[D1]]{{\]}}
// CHECK-SAME: variables_space_order = #[[MAP]]
// CHECK-SAME: variables_space_set = #[[SET2]]
// CHECK-SAME: : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
// CHECK-SAME: -> !ktdp.access_tile<32x2x64xindex>
func.func @mixed_iab_subscripts(
    %base : memref<64x2x64xf16>,
    %iab  : memref<2x32xindex, "IAB">,
    %c0   : index,
    %i1   : index,       // outer SSA value (e.g. scf.for IV selecting IAB row)
    %arg6 : index,       // intermediate variable selecting IAB column
    %arg7 : index,
    %arg8 : index) {
  %tile = ktdp_lowering.construct_indirect_access_tile
      intermediate_variables(%arg6, %arg7, %arg8)
      base_ptr = %iab[%i1, %arg6]
      %base[%c0, %arg7, %arg8]
      {variables_space_set = #set2, variables_space_order = #map}
      : memref<64x2x64xf16>, memref<2x32xindex, "IAB">
      -> !ktdp.access_tile<32x2x64xindex>
  return
}
