// RUN: dataflow-scheduler-opt --fuse-linalg %s | FileCheck %s

// Tests that a mapping implied by the contents of a `linalg.generic` is
// propagated onto the operation itself.

#id = affine_map<(d0) -> (d0)>

// A payload op that is already inside a generic implies the same mapping. This
// is the result of --convert-elementwise-to-linalg.
// CHECK-LABEL:   func.func @implied_from_payload
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "SFU"
func.func @implied_from_payload(%a: tensor<8xf32>, %b: tensor<8xf32>)
    -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y {ktdf_arch.maps_to = "SFU"} : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// Payload ops that require different resources imply all of them at once.
// CHECK-LABEL:   func.func @implied_union
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = ["SFU", "PE"]
func.func @implied_union(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y {ktdf_arch.maps_to = "SFU"} : f32
    %s = math.sqrt %m {ktdf_arch.maps_to = "PE"} : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// An explicitly declared mapping is authoritative and is not widened by what
// the payload implies.
// CHECK-LABEL:   func.func @explicit_mapping_is_kept
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "PE"
// CHECK-NOT:         ktdf_arch.maps_to = ["PE", "SFU"]
func.func @explicit_mapping_is_kept(%a: tensor<8xf32>, %b: tensor<8xf32>)
    -> tensor<8xf32> {
  %e = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%e : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = "PE"} {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y {ktdf_arch.maps_to = "SFU"} : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
