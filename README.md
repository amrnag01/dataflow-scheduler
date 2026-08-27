# Dataflow Scheduler 

## Prerequisites

- **Python 3.10-3.13** (3.12 recommended)
- **LLVM/MLIR `llvmorg-22.1.3`** with Python bindings
- **Git** and **CMake >= 3.21.0** (for building LLVM/MLIR)
- **Anthropic API Key** for agentic tile size selection
- **SAMM-KTDF** cost model for latency evaluation

## Installation and Build

### Step 1: Build LLVM and MLIR (if not already installed)

```sh
# Clone the repository as lean as possible
git clone https://github.com/llvm/llvm-project.git \
  --branch llvmorg-22.1.3 --sparse --depth 1
cd llvm-project
git sparse-checkout add cmake libc llvm mlir runtimes third-party

# Configure LLVM & MLIR (with Python bindings)
cmake -S ./llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_PYTHON_STUBGEN_ENABLED=ON

# Build LLVM (this may take a while)
cmake --build build
```

### Step 2: Build Dataflow Scheduler with Python Bindings

```sh
# Clone the dataflow scheduler with submodules
git clone --recursive https://github.com/amrnag01/dataflow-scheduler.git
cd dataflow-scheduler

# Configure with Python bindings enabled
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=<path-to-llvm-build>/lib/cmake/mlir \
  -DDataflowScheduler_ENABLE_PYTHON_BINDINGS=ON

# Build
cmake --build build --target all
```

Replace `<path-to-llvm-build>` with the actual path to your LLVM build directory.

The Python bindings will be located at:
```
dataflow-scheduler/build/python_packages/scheduler
```

### Step 3: Set Up SAMM-KTDF Cost Model

```sh
# Clone samm-ktdf (Note: recommended to download samm-ktdf to a different folder, separate from dataflow-scheduler)
git clone https://github.ibm.com/SAMM-toolchain/samm-ktdf.git
cd samm-ktdf

# Create a Python virtual environment for the cost model
python3 -m venv samm_env

# Activate the virtual environment
source samm_env/bin/activate

# Install SAMM Core
pip3 install samm-core@git+ssh://git@github.ibm.com/SAMM-toolchain/samm-core.git
```

Always activate the SAMM venv before running the dataflow scheduler:
```sh
source <path-to-samm-ktdf>/samm_env/bin/activate
```

## Running Dataflow Scheduler with Agentic Tile Size Selection

### Basic Command

```sh
# Activate the SAMM venv
source <path-to-samm-ktdf>/samm_env/bin/activate

# Run the dataflow scheduler
./build/bin/dataflow-scheduler -kEmitDFIR \
  -device=<path-to-device-spec> \
  -ktdf_bindings_dir dataflow-scheduler/build/python_packages/scheduler \
  -cost_model_path <path-to-samm-ktdf> \
  -anthropic-api-key <your-api-key> \
  <input-mlir-file> \
  [options]
```

## Command Line Options

### Required Options

| Option | Description |
|--------|-------------|
| `-kEmitDFIR` | Emit DataflowIR (required for code generation) |
| `-device=<file>` | Path to device architecture specification (MLIR file describing hardware target) |
| `-ktdf_bindings_dir <path>` | Path to MLIR Python bindings directory (`<dataflow-scheduler-build>/python_packages/scheduler`) |
| `-cost_model_path <path>` | Path to samm-ktdf cost model directory |
| `-anthropic-api-key <key>` | Anthropic API key for agentic tile size selection (or set `ANTHROPIC_API_KEY` environment variable) |

### Optional Options

| Option | Description |
|--------|-------------|
| `-agent-debug` | Enable debug mode: dump all IRs passed to cost model in `debug/success/` and `debug/fail/` directories with naming `ktdf_<tile_size_1>_<tile_size_2>.mlir` |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (alternative to `-anthropic-api-key` flag) |

