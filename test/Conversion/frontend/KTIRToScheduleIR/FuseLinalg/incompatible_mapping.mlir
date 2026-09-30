// RUN: dataflow-scheduler-opt --fuse-linalg %s | FileCheck %s
// RUN: dataflow-scheduler-opt --fuse-linalg %s --relax-fuse-linalg | FileCheck %s --check-prefix=RELAX

// Tests that operations which would otherwise be fused are kept apart when
// their mappings refer to different resources.

#id = affine_map<(d0) -> (d0)>

// CHECK-LABEL:   func.func @explicit_conflict
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "SFU"
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "PE"
// CHECK-NOT:       linalg.generic
// RELAX-LABEL:   func.func @explicit_conflict
// RELAX:           linalg.generic
// RELAX-SAME:        ktdf_arch.maps_to = ["PE", "SFU"]
// RELAX-NOT:       linalg.generic
func.func @explicit_conflict(%a: tensor<8xf32>, %b: tensor<8xf32>)
    -> tensor<8xf32> {
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
       attrs = {ktdf_arch.maps_to = "PE"} {
  ^bb0(%x: f32, %o: f32):
    %s = math.sqrt %x : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// The conflict is detected on mappings that are only implied by the payloads,
// which means propagation has to happen before fusion is attempted.
// CHECK-LABEL:   func.func @implied_conflict
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "SFU"
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "PE"
// CHECK-NOT:       linalg.generic
// RELAX-LABEL:   func.func @implied_conflict
// RELAX:           linalg.generic
// RELAX-SAME:        ktdf_arch.maps_to = ["PE", "SFU"]
// RELAX-NOT:       linalg.generic
func.func @implied_conflict(%a: tensor<8xf32>, %b: tensor<8xf32>)
    -> tensor<8xf32> {
  %e0 = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%e0 : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y {ktdf_arch.maps_to = "SFU"} : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  %e1 = tensor.empty() : tensor<8xf32>
  %1 = linalg.generic {indexing_maps = [#id, #id],
                       iterator_types = ["parallel"]}
       ins(%0 : tensor<8xf32>) outs(%e1 : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %s = math.sqrt %x {ktdf_arch.maps_to = "PE"} : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL:   func.func @unmapped_producer
// CHECK:           linalg.generic
// CHECK:             arith.mulf
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "SFU"
// CHECK:             math.sqrt
// RELAX-LABEL:   func.func @unmapped_producer
// RELAX:           linalg.generic
// RELAX-SAME:        ktdf_arch.maps_to = "SFU"
// RELAX-NOT:       linalg.generic
func.func @unmapped_producer(%a: tensor<8xf32>, %b: tensor<8xf32>)
    -> tensor<8xf32> {
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
       ins(%0 : tensor<8xf32>) outs(%e1 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %o: f32):
    %s = math.sqrt %x : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-LABEL:   func.func @unmapped_consumer
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "SFU"
// CHECK:             arith.mulf
// CHECK:           linalg.generic
// CHECK:             math.sqrt
// RELAX-LABEL:   func.func @unmapped_consumer
// RELAX:           linalg.generic
// RELAX-SAME:        ktdf_arch.maps_to = "SFU"
// RELAX-NOT:       linalg.generic
func.func @unmapped_consumer(%a: tensor<8xf32>, %b: tensor<8xf32>)
    -> tensor<8xf32> {
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
       ins(%0 : tensor<8xf32>) outs(%e1 : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %s = math.sqrt %x : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
