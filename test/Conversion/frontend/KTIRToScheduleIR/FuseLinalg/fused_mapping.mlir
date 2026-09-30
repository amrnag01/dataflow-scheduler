// RUN: dataflow-scheduler-opt --fuse-linalg %s | FileCheck %s

// Tests that a fused operation carries the mapping of the operations it was
// fused from. Fusion only happens between compatible mappings, so the result
// never has to combine two different mappings.

#id = affine_map<(d0) -> (d0)>

// CHECK-LABEL:   func.func @same_mapping
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "SFU"
// CHECK:             arith.mulf
// CHECK:             math.sqrt
// CHECK-NOT:       linalg.generic
func.func @same_mapping(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %e0 = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%e0 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  %e1 = tensor.empty() : tensor<8xf32>
  %1 = linalg.generic {indexing_maps = [#id, #id],
                       iterator_types = ["parallel"]}
       ins(%0 : tensor<8xf32>) outs(%e1 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %o: f32):
    %s = math.sqrt %x : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// Nothing is mapped, so the fused op stays unmapped.
// CHECK-LABEL:   func.func @no_mapping
// CHECK:           linalg.generic
// CHECK-NOT:         ktdf_arch.maps_to
// CHECK:             arith.mulf
// CHECK:             math.sqrt
// CHECK-NOT:       linalg.generic
func.func @no_mapping(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %e0 = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%e0 : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  %e1 = tensor.empty() : tensor<8xf32>
  %1 = linalg.generic {indexing_maps = [#id, #id],
                       iterator_types = ["parallel"]}
       ins(%0 : tensor<8xf32>) outs(%e1 : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %s = math.sqrt %x : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL:   func.func @compatible_mapping
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = ["PE", "SFU"]
// CHECK:             arith.mulf
// CHECK:             math.sqrt
// CHECK-NOT:       linalg.generic
func.func @compatible_mapping(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %e0 = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%e0 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = ["SFU", "PE"]} {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  %e1 = tensor.empty() : tensor<8xf32>
  %1 = linalg.generic {indexing_maps = [#id, #id],
                       iterator_types = ["parallel"]}
       ins(%0 : tensor<8xf32>) outs(%e1 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = ["PE", "SFU"]} {
  ^bb0(%x: f32, %o: f32):
    %s = math.sqrt %x : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL:   func.func @multiple_candidates
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "SFU"
// CHECK-DAG:         arith.negf
// CHECK-DAG:         math.sqrt
// CHECK:             arith.addf
// CHECK-NOT:       linalg.generic
func.func @multiple_candidates(%a: tensor<8xf32>, %b: tensor<8xf32>) -> tensor<8xf32> {
  %e0 = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<8xf32>) outs(%e0 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %o: f32):
    %s = arith.negf %x : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  %e1 = tensor.empty() : tensor<8xf32>
  %1 = linalg.generic {indexing_maps = [#id, #id],
                       iterator_types = ["parallel"]}
       ins(%b : tensor<8xf32>) outs(%e1 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %o: f32):
    %s = math.sqrt %x : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  %e2 = tensor.empty() : tensor<8xf32>
  %2 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%0, %1 : tensor<8xf32>, tensor<8xf32>) outs(%e2 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %s = arith.addf %x, %y : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %2 : tensor<8xf32>
}
