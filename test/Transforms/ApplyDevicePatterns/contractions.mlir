// RUN: dataflow-scheduler-opt %s -apply-device-patterns='groups=pre_mapping' | FileCheck %s

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#contraction_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  
  // CHECK-LABEL: func.func @matmul_ep_1core
  func.func @matmul_ep_1core() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<1x64x64xf16>

    %a_base = arith.constant 0 : index
    %w_base = arith.constant 8589934592 : index
    %c_base = arith.constant 17179869184 : index

    %a_view = ktdp.construct_memory_view %a_base, sizes: [4, 64, 64], strides: [4096, 64, 1] {
      coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<4x64x64xf16>

    %w_view = ktdp.construct_memory_view %w_base, sizes: [1, 4, 64, 64], strides: [16384, 4096, 64, 1] {
      coordinate_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 4 >= 0, d2 >= 0, -d2 + 64 >= 0, d3 >= 0, -d3 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<1x4x64x64xf16>

    %c_view = ktdp.construct_memory_view %c_base, sizes: [1, 64, 64], strides: [4096, 64, 1] {
      coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<1x64x64xf16>

    %a_tile = ktdp.construct_access_tile %a_view[%c0, %c0, %c0] {
      access_tile_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    } : memref<4x64x64xf16> -> !ktdp.access_tile<4x64x64xindex>
    %a_tensor = ktdp.load %a_tile : !ktdp.access_tile<4x64x64xindex> -> tensor<4x64x64xf16>

    %w_tile = ktdp.construct_access_tile %w_view[%c0, %c0, %c0, %c0] {
      access_tile_set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 4 >= 0, d2 >= 0, -d2 + 64 >= 0, d3 >= 0, -d3 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    } : memref<1x4x64x64xf16> -> !ktdp.access_tile<1x4x64x64xindex>
    %w_tensor = ktdp.load %w_tile : !ktdp.access_tile<1x4x64x64xindex> -> tensor<1x4x64x64xf16>

    %a_via_l0 = ktdf.via["L0"] %a_tensor : tensor<4x64x64xf16>

    // CHECK: linalg.generic
    // CHECK-SAME: ktdf_arch.maps_to = "MPE_ARRAY"
    %c_acc = linalg.generic {
        indexing_maps = [
          affine_map<(out, batch, in, in2, out2) -> (in, batch, in2)>,
          affine_map<(out, batch, in, in2, out2) -> (out, in, in2, out2)>,
          affine_map<(out, batch, in, in2, out2) -> (out, batch, out2)>
        ],
        iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel"]
      } ins(%a_via_l0, %w_tensor : tensor<4x64x64xf16>, tensor<1x4x64x64xf16>)
        outs(%cst : tensor<1x64x64xf16>) {
    ^bb0(%a: f16, %w: f16, %c: f16):
      %mul = arith.mulf %a, %w : f16
      %acc = arith.addf %mul, %c : f16
      linalg.yield %acc : f16
    } -> tensor<1x64x64xf16>

    %c_tile = ktdp.construct_access_tile %c_view[%c0, %c0, %c0] {
      access_tile_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    } : memref<1x64x64xf16> -> !ktdp.access_tile<1x64x64xindex>
    ktdp.store %c_acc, %c_tile : tensor<1x64x64xf16>, !ktdp.access_tile<1x64x64xindex>

    return
  }
}
