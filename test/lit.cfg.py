#
# Copyright 2026 The KTIR Scheduler Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import lit.formats
import lit.llvm

llvm_config = lit.llvm.llvm_config

config.name = "DataflowScheduler"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.dataflow_scheduler_obj_root, "test")

config.substitutions.extend(
    [
        ("%PATH%", config.environment["PATH"]),
        ("%shlibext", config.llvm_shlib_ext),
        ("%test_source_root", config.test_source_root),
        ("%test_exec_root", config.test_exec_root),
    ]
)

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "CMakeLists.txt"]

tools = ["dataflow-scheduler", "dataflow-scheduler-opt", "FileCheck", "not"]
tool_dirs = [config.dataflow_scheduler_tools_dir, config.llvm_tools_dir]

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

python_path = [
    os.path.join(
        config.dataflow_scheduler_obj_root, config.python_package_install_prefix, ".."
    )
]
if "PYTHONPATH" in os.environ:
    python_path += [os.environ["PYTHONPATH"]]
llvm_config.with_environment("PYTHONPATH", python_path, append_path=True)

llvm_config.add_tool_substitutions(tools, tool_dirs)
