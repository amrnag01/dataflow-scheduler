cmake_minimum_required(VERSION 3.21)

# Add custom modules to the search path.
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(DataflowScheduler 
  VERSION 0.1
  LANGUAGES C CXX
)

# Set defaults for compiler features and options.
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

if(PROJECT_IS_TOP_LEVEL)
  # Find MLIR.
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  # Find external dependencies for agent integration
  find_package(CURL REQUIRED)

  # Fetch nlohmann_json
  include(FetchContent)
  FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(json)

  # Load LLVM and MLIR CMake modules.
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  # Apply target properties for LLVM/MLIR globally.
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
  link_directories(${LLVM_BUILD_LIBRARY_DIR})
  add_definitions(${LLVM_DEFINITIONS})

  # Set shared output directories.
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${PROJECT_BINARY_DIR}/lib)
  set(LLVM_RUNTIME_OUTPUT_INTDIR ${PROJECT_BINARY_DIR}/bin)

  # Set up RPATH for installed distribution.
  include(GNUInstallDirs)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH YES)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

#
# Configure options
#

option(DataflowScheduler_ENABLE_PYTHON_BINDINGS "Build Python bindings" NO)
option(DataflowScheduler_BUILD_TOOLS "Build tool executables" ${PROJECT_IS_TOP_LEVEL})
option(DataflowScheduler_ENABLE_WERROR "Treat compiler warnings as errors (-Werror)" YES)

#
# Umbrella targets
#

# Build all CAPI libraries.
add_custom_target(dataflow-scheduler-capi)
# Build all documentation targets.
add_custom_target(dataflow-scheduler-doc)
# Build all generated include targets.
add_custom_target(dataflow-scheduler-headers)

if(NOT TARGET check-all)
  # Run all regression tests.
  add_custom_target(check-all)
endif()

#
# Derived variables
#

# Handle linking against LLVM/MLIR shared library.
if(TARGET MLIR AND NOT DEFINED MLIR_LINK_MLIR_DYLIB)
  # As a workaround for MLIRConfig.cmake not exporting this variable, we will
  # infer that it is set if the DSO target exists.
  set(MLIR_LINK_MLIR_DYLIB YES)
endif()
if(LLVM_LINK_LLVM_DYLIB)
  get_target_property(_llvm_dso LLVM LOCATION)
  message(STATUS "Linking against: ${_llvm_dso}")
endif()
if(MLIR_LINK_MLIR_DYLIB)
  get_target_property(_mlir_dso MLIR LOCATION)
  message(STATUS "Linking against: ${_mlir_dso}")
endif()

# Handle building combined Python bindings.
if(DataflowScheduler_ENABLE_PYTHON_BINDINGS)
  set(MLIR_ENABLE_BINDINGS_PYTHON ON)
  set(KTIR_ENABLE_PYTHON_BINDINGS ON)
  set(DataflowSchedulerDialects_ENABLE_PYTHON_BINDINGS ON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()

  if(NOT DEFINED MLIR_PYTHON_PACKAGE_PREFIX)
    set(MLIR_PYTHON_PACKAGE_PREFIX "mlir_scheduler" CACHE STRING "" FORCE)
  endif()
  if(NOT DEFINED MLIR_BINDINGS_PYTHON_NB_DOMAIN)
    set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "mlir_scheduler" CACHE STRING "" FORCE)
  endif()
  if(NOT DEFINED MLIR_BINDINGS_PYTHON_INSTALL_PREFIX)
    set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX 
      "python_packages/scheduler/${MLIR_PYTHON_PACKAGE_PREFIX}" 
      CACHE STRING "" FORCE
    )
  endif()
endif()

add_subdirectory(external)

target_link_libraries(DataflowSchedulerDialectsKTDFArchTransforms PUBLIC
  KTIR::KtdpDialect
  KTIR::SpyreOpDialect
)
target_compile_definitions(DataflowSchedulerDialectsKTDFArchTransforms PRIVATE
  DATAFLOW_SCHEDULER_ENABLE_KTIR
)
# add_mlir_library(ENABLE_AGGREGATION) internally creates an OBJECT library
# target named obj.<name> where the .cpp files are actually compiled. The
# target_compile_definitions above only reaches the interface target, so we
# must repeat the define on the OBJECT target to make
# #ifdef DATAFLOW_SCHEDULER_ENABLE_KTIR active in the translation units.
target_compile_definitions(obj.DataflowSchedulerDialectsKTDFArchTransforms PRIVATE
  DATAFLOW_SCHEDULER_ENABLE_KTIR
)

add_dependencies(dataflow-scheduler-headers dataflow-scheduler-dialects-headers ktir-headers)
add_dependencies(dataflow-scheduler-doc dataflow-scheduler-dialects-doc ktir-doc)

# Treat warnings as errors for all first-party sources. This block is
# intentionally placed after add_subdirectory(external) so that the flag is
# not applied to third-party submodules.
if(DataflowScheduler_ENABLE_WERROR)
  add_compile_options(-Werror)
  message(STATUS "DataflowScheduler: -Werror enabled for first-party sources")
endif()

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# Include DataflowScheduler CMake module.
set(DataflowScheduler_INCLUDE_DIRS 
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  CACHE STRING "" FORCE
)
include(AddDataflowScheduler)
include(clang-format)

add_subdirectory(include)
add_subdirectory(lib)
if(DataflowScheduler_ENABLE_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()
add_subdirectory(test)
add_subdirectory(tools)

# Install instructions, these must come last so that all properties are set.
add_subdirectory(cmake)
