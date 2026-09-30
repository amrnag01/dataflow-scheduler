//===------------------------------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_MEMORYTRACKER_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_MEMORYTRACKER_H_

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Error.h>

#include <cstddef>

#include "dataflow-scheduler/Analysis/Mapping.h"

namespace scheduler {

// Forward declaration
namespace arch_view {
class MemoryTree;
}

/// @brief MemoryTracker manages buffer allocations for different memory spaces.
/// It tracks the next available address for each memory space and validates
/// that allocations fit within the available capacity.
///
/// The tracker is initialized from a MemoryTree view which provides capacity
/// information for each memory resource.
class MemoryTracker {
 public:
  /// @brief Construct a MemoryTracker from a MemoryTree
  /// @param memory_tree The memory hierarchy view containing capacity info
  explicit MemoryTracker(const arch_view::MemoryTree& memory_tree);

  /// @brief Allocate memory and return the assigned address
  /// @param memory_resource The memory resource attribute (e.g., L1)
  /// @param size_in_bytes Size of the allocation in bytes
  /// @param alignment Byte alignment requirement (default: 1)
  /// @return The allocated address on success, or an error if allocation fails
  llvm::Expected<size_t> allocate(ResourceType memory_resource,
                                  size_t size_in_bytes, size_t alignment = 1);

  /// @brief Get the current next available address for a memory resource
  /// @param memory_resource The memory resource attribute to query
  /// @return The next available address; asserts if the resource is unknown
  size_t getNextAvailableAddress(ResourceType memory_resource) const;

  /// @brief Get the total allocated size for a memory resource
  /// @param memory_resource The memory resource attribute to query
  /// @return Total bytes allocated in this memory resource; asserts if unknown
  size_t getTotalAllocated(ResourceType memory_resource) const;

  /// @brief Puts each of \p memory_resources back to its first address
  /// @param memory_resources The memory resources to make available again
  void reset(llvm::ArrayRef<ResourceType> memory_resources);

 private:
  /// Memory resource capacities (total bytes per resource)
  llvm::DenseMap<ResourceType, size_t> capacities_;

  /// Reserved bytes per resource — first allocatable address after reset
  llvm::DenseMap<ResourceType, size_t> reserved_;

  /// Track next available address per memory resource attribute
  llvm::DenseMap<ResourceType, size_t> next_address_;

  /// Helper to align an address to the specified alignment
  static size_t alignAddress(size_t address, size_t alignment);

  /// Helper to produce a printable string for a memory resource
  static std::string resourceToString(ResourceType memory_resource);
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_MEMORYTRACKER_H_

// Made with Bob
