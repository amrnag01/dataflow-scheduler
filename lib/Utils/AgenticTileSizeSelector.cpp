//===-- AgenticTileSizeSelector.cpp -------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Utils/AgenticTileSizeSelector.h"

#include <curl/curl.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#include <nlohmann/json.hpp>
#pragma clang diagnostic pop

#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <sstream>

#include "dataflow-scheduler/Dialect/KTDF/TileSizeApply.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace scheduler {

using json = nlohmann::json;

// CURL write callback for response body
static size_t WriteCallback(void* contents, size_t size, size_t nmemb,
                            std::string* s) {
  s->append((char*)contents, size * nmemb);
  return size * nmemb;
}

AgenticTileSizeSelector::AgenticTileSizeSelector(
    const std::string& api_key, const std::string& ktdf_bindings_dir,
    const std::string& cost_model_path, bool debug)
    : api_key_(api_key),
      ktdf_bindings_dir_(ktdf_bindings_dir),
      cost_model_path_(cost_model_path),
      debug_(debug),
      num_tile_sizes_(0) {}

AgenticTileSizeSelector::~AgenticTileSizeSelector() = default;

bool AgenticTileSizeSelector::generateSymbolicCostModel(
    mlir::ModuleOp module, const std::string& ir_str,
    llvm::ArrayRef<TileSizeInfo> analyses) {
  // Write IR to temp file
  llvm::SmallString<256> temp_file;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("symbolic_cost", "mlir", temp_file);
  if (ec) {
    llvm::errs() << "Failed to create temp file for symbolic cost model\n";
    return false;
  }

  std::ofstream f(temp_file.c_str());
  f << ir_str;
  f.close();

  // Invoke samm-ktdf to generate symbolic cost model (automatically generated
  // when tile sizes are unresolved)
  std::string cmd = "cd '" + cost_model_path_ +
                    "' && source samm_env/bin/activate && python3.12 main.py "
                    "--mlir-bindings-dir '" +
                    ktdf_bindings_dir_ + "' --input-file '" +
                    std::string(temp_file.c_str()) + "'";

  std::string output_file;
  llvm::SmallString<256> temp_out;
  ec = llvm::sys::fs::createTemporaryFile("symbolic_out", "txt", temp_out);
  if (ec) {
    llvm::sys::fs::remove(temp_file);
    return false;
  }
  output_file = std::string(temp_out.c_str());

  // Execute command
  std::string full_cmd = cmd + " > " + output_file + " 2>&1";
  int ret_code = system(full_cmd.c_str());

  // Read output
  std::string output_content;
  std::ifstream output_stream(output_file.c_str());
  if (output_stream.is_open()) {
    output_content =
        std::string((std::istreambuf_iterator<char>(output_stream)),
                    std::istreambuf_iterator<char>());
    output_stream.close();
  }

  llvm::sys::fs::remove(output_file);
  llvm::sys::fs::remove(temp_file);

  if (ret_code != 0) {
    llvm::errs() << "Symbolic cost model generation failed:\n"
                 << output_content << "\n";
    return false;
  }

  // Check if model.py was generated
  std::string model_py_path = cost_model_path_ + "/latency_model/model.py";
  std::ifstream model_file(model_py_path);
  if (!model_file.is_open()) {
    llvm::errs() << "model.py not generated at " << model_py_path << "\n";
    return false;
  }

  // Read the generated model.py
  std::stringstream model_buffer;
  model_buffer << model_file.rdbuf();
  std::string original_function = model_buffer.str();
  model_file.close();

  // Store the cost function for display in the prompt (don't modify it)
  symbolic_cost_function_ = original_function;

  // Determine number of tile sizes from function signature
  // Extract only s0, s1, ... parameters, excluding use_locals
  std::regex sig_regex(R"(def\s+latency_function\s*\(\s*([^)]+)\s*\))");
  std::smatch match;
  if (std::regex_search(original_function, match, sig_regex)) {
    std::string params = match[1];
    // Remove "use_locals=False" or similar keyword arguments
    std::regex kwarg_regex(",?\\s*use_locals\\s*=[^,)]*");
    params = std::regex_replace(params, kwarg_regex, "");
    // Count commas and add 1 to get tile size count
    num_tile_sizes_ = std::count(params.begin(), params.end(), ',') + 1;
  } else {
    llvm::errs() << "Could not parse latency_function signature\n";
    return false;
  }

  return true;
}

std::string AgenticTileSizeSelector::instrumentPythonCostFunction(
    const std::string& original_function) {
  // Find the return statement and modify it to return a dict with all variables
  std::string instrumented = original_function;

  // Replace "return p_0" with a dict return that includes all locals
  std::string old_return = "    return p_0";
  std::string new_return = R"(    result_vars = {}
    for var_name in dir():
        if var_name.startswith('x') or var_name == 'p_0':
            try:
                result_vars[var_name] = locals()[var_name]
            except:
                pass
    return {"latency": p_0, "variables": result_vars})";

  size_t pos = instrumented.find(old_return);
  if (pos != std::string::npos) {
    instrumented.replace(pos, old_return.length(), new_return);
  }

  return instrumented;
}

AgenticTileSizeSelector::CostEvaluation
AgenticTileSizeSelector::evaluateSymbolicCostFunction(
    const std::vector<int64_t>& tile_sizes) {
  if (tile_sizes.size() != (size_t)num_tile_sizes_) {
    return {false, 0.0, {}, "Tile size count mismatch"};
  }

  // Create a temporary Python script that imports and calls the function
  llvm::SmallString<256> temp_py;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("eval_cost", "py", temp_py);
  if (ec) {
    return {false, 0.0, {}, "Failed to create temp Python file"};
  }

  std::ofstream py_file(temp_py.c_str());
  py_file << "import sys\nimport json\n";
  py_file << "sys.path.insert(0, '" << cost_model_path_ << "/latency_model')\n";
  py_file << "from model import latency_function\n";
  py_file << "result = latency_function(";
  for (size_t i = 0; i < tile_sizes.size(); ++i) {
    if (i > 0) py_file << ", ";
    py_file << tile_sizes[i];
  }
  py_file << ", use_locals=True)\n";
  py_file << "latency, all_locals = result\n";
  py_file << "# Extract only numeric intermediate variables (x0, x1, ... and "
             "p_0)\n";
  py_file << "vars_dict = {}\n";
  py_file << "for name, value in all_locals.items():\n";
  py_file << "    if isinstance(value, (int, float)) and (name.startswith('x') "
             "or name == 'p_0'):\n";
  py_file << "        vars_dict[name] = float(value)\n";
  py_file << "output = {'latency': float(latency), 'variables': vars_dict}\n";
  py_file << "print(json.dumps(output))\n";
  py_file.close();

  // Execute the Python script
  std::string cmd = "cd '" + cost_model_path_ +
                    "' && source samm_env/bin/activate && python3.12 " +
                    std::string(temp_py.c_str());

  std::string output_file;
  llvm::SmallString<256> temp_out;
  ec = llvm::sys::fs::createTemporaryFile("eval_out", "txt", temp_out);
  if (ec) {
    llvm::sys::fs::remove(temp_py);
    return {false, 0.0, {}, "Failed to create output temp file"};
  }
  output_file = std::string(temp_out.c_str());

  std::string full_cmd = cmd + " > " + output_file + " 2>&1";
  int ret_code = system(full_cmd.c_str());

  // Read output
  std::string output_content;
  std::ifstream output_stream(output_file.c_str());
  if (output_stream.is_open()) {
    output_content =
        std::string((std::istreambuf_iterator<char>(output_stream)),
                    std::istreambuf_iterator<char>());
    output_stream.close();
  }

  llvm::sys::fs::remove(temp_py);
  llvm::sys::fs::remove(output_file);

  if (ret_code != 0) {
    return {
        false, 0.0, {}, "Cost function evaluation failed: " + output_content};
  }

  // Parse JSON output to extract latency and variables
  std::map<std::string, double> variables;
  double latency = 0.0;

  // Validate JSON format first
  if (!json::accept(output_content)) {
    return {false, 0.0, {}, "Failed to parse JSON output: " + output_content};
  }

  json result_json = json::parse(output_content);
  if (result_json.contains("latency")) {
    latency = result_json["latency"].get<double>();
  }
  if (result_json.contains("variables") &&
      result_json["variables"].is_object()) {
    for (auto& [key, val] : result_json["variables"].items()) {
      if (val.is_number()) {
        variables[key] = val.get<double>();
      }
    }
  }

  return {true, latency, variables, ""};
}

std::vector<int64_t> AgenticTileSizeSelector::run(
    mlir::ModuleOp module, llvm::ArrayRef<TileSizeInfo> analyses) {
  if (analyses.empty()) {
    return {};
  }

  // Clean up debug folder if it exists and recreate it
  if (debug_) {
    llvm::sys::fs::remove_directories("debug");
    llvm::sys::fs::create_directories("debug/success");
    llvm::sys::fs::create_directories("debug/fail");
  }

  // Compute loop granularities once
  computeLoopGranularities(analyses);

  // Print IR to string for symbolic cost model generation
  std::string ir_str;
  llvm::raw_string_ostream ir_stream(ir_str);
  module.print(ir_stream);
  ir_stream.flush();

  // Generate symbolic cost model from original IR
  llvm::errs() << "[Agent] Generating symbolic cost model...\n";
  if (!generateSymbolicCostModel(module, ir_str, analyses)) {
    llvm::report_fatal_error("Failed to generate symbolic cost model");
  }
  llvm::errs() << "[Agent] Symbolic cost model generated with "
               << num_tile_sizes_ << " tile size parameters\n";

  std::string system_prompt = buildSystemPrompt(analyses);
  std::string tool_schemas = buildToolSchemas();

  std::vector<json> messages;
  json user_msg;
  user_msg["role"] = "user";
  user_msg["content"] =
      "Optimize this symbolic cost function by PURE MATHEMATICAL ANALYSIS.\n\n"
      "Do NOT try random tile sizes. Every call to evaluate_cost must be "
      "because "
      "you mathematically predict it will improve on the best result so "
      "far.\n\n"
      "IMPORTANT: (1) Your reasoning must be grounded ONLY in mathematical "
      "analysis of "
      "the cost function AND intermediate variable values returned to you. As "
      "part of the reasoning, look at the dominant intermediate values "
      "returned to you, and see how they changed from your best solution. (2) "
      "You are allowed to call the evaluate_cost tool "
      "only "
      "if you believe that your newly selected tile sizes for evaluation will "
      "be better than the best solution you have found so far \n\n";
  messages.push_back(user_msg);

  // Tool-use loop
  int max_iterations = 20;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    // Build messages JSON array
    json messages_array = json::array();
    for (const auto& msg : messages) {
      messages_array.push_back(msg);
    }

    std::string response =
        makeHttpRequestWithTools(system_prompt, messages_array, tool_schemas);

    // Parse response for tool use
    json response_json;
    response_json = json::parse(response);

    if (response_json.contains("error")) {
      llvm::errs() << "[Agent] Error in response: "
                   << response_json["error"].dump() << "\n";
    }

    if (response_json.contains("content")) {
      auto content = response_json["content"];
      if (!content.is_array()) {
        llvm::report_fatal_error("LLM response content is not an array");
      }

      bool found_tool_use = false;

      for (const auto& block : content) {
        if (block.contains("type") && block["type"] == "tool_use") {
          found_tool_use = true;
          std::string tool_name = block["name"];

          if (tool_name == "submit_final_answer") {
            // Extract final answer
            json input = block["input"];
            std::vector<int64_t> result;

            // Parse tile_sizes as direct integers (s0, s1, ...)
            for (const auto& tile_size : input["tile_sizes"]) {
              if (tile_size.is_number()) {
                result.push_back(tile_size.get<int64_t>());
              }
            }

            if (result.size() != (size_t)num_tile_sizes_) {
              llvm::report_fatal_error(llvm::Twine("Final answer has ") +
                                       std::to_string(result.size()) +
                                       " tile sizes but expected " +
                                       std::to_string(num_tile_sizes_));
            }

            // Validate each tile size (map s0, s1, ... to analysis indices)
            for (size_t i = 0; i < result.size() && i < analyses.size(); ++i) {
              auto& analysis = const_cast<TileSizeInfo&>(analyses[i]);
              int64_t tile_size = result[i];
              int64_t min_value =
                  analysis.reserve_size_op.getMinValue().getSExtValue();
              int64_t divisibility =
                  analysis.reserve_size_op.getDivisibility().getSExtValue();

              if (tile_size < min_value) {
                llvm::report_fatal_error(
                    llvm::Twine("Final tile size s") + std::to_string(i) +
                    " is " + std::to_string(tile_size) + " but minimum is " +
                    std::to_string(min_value));
              }

              if (tile_size % divisibility != 0) {
                llvm::report_fatal_error(llvm::Twine("Final tile size s") +
                                         std::to_string(i) + " is " +
                                         std::to_string(tile_size) +
                                         " but must be divisible by " +
                                         std::to_string(divisibility));
              }

              // Check granularity constraint
              int64_t granularity = 1;
              for (const auto& loop_info : analysis.associated_loops) {
                auto it = loop_granularities_.find(loop_info.loop);
                if (it != loop_granularities_.end()) {
                  granularity = it->second;
                  break;
                }
              }
              if (granularity > 1 && tile_size % granularity != 0) {
                llvm::report_fatal_error(
                    llvm::Twine("Final tile size s") + std::to_string(i) +
                    " is " + std::to_string(tile_size) +
                    " but must be divisible by " + std::to_string(granularity) +
                    " (parallel region constraint)");
              }
            }

            llvm::errs() << "[Agent] Selected tile sizes: ";
            for (size_t i = 0; i < result.size(); ++i) {
              llvm::errs() << (i > 0 ? ", " : "") << result[i];
            }
            llvm::errs() << "\n";
            if (input.contains("explanation")) {
              llvm::errs() << "[Agent] Reasoning: "
                           << input["explanation"].get<std::string>() << "\n";
            }

            return result;
          } else if (tool_name == "evaluate_cost") {
            // Execute symbolic cost evaluation
            json input = block["input"];
            std::vector<int64_t> tile_sizes;

            // Parse tile_sizes as direct integers
            for (const auto& ts : input["tile_sizes"]) {
              if (ts.is_number()) {
                tile_sizes.push_back(ts.get<int64_t>());
              }
            }

            std::string reasoning = input["reasoning"].get<std::string>();

            // Print what we're trying
            llvm::errs() << "[Agent] Evaluating: ";
            for (size_t i = 0; i < tile_sizes.size(); ++i) {
              llvm::errs() << (i > 0 ? ", " : "") << tile_sizes[i];
            }
            llvm::errs() << "\n";
            llvm::errs() << "Reasoning: " << reasoning << "\n";

            auto eval_result = evaluateSymbolicCostFunction(tile_sizes);

            if (eval_result.success) {
              std::ostringstream latency_str;
              latency_str << std::setprecision(15) << eval_result.latency;
              llvm::errs() << "Latency: " << latency_str.str() << " sec\n";
            } else {
              llvm::errs() << "Error: " << eval_result.error_message << "\n";
            }

            // Add assistant message with tool_use
            json assistant_msg;
            assistant_msg["role"] = "assistant";
            json content_array = json::array();
            content_array.push_back(block);
            assistant_msg["content"] = content_array;
            messages.push_back(assistant_msg);

            // Add tool result to messages for next iteration
            json tool_result_msg;
            tool_result_msg["role"] = "user";
            json user_content = json::array();

            json tool_result;
            tool_result["type"] = "tool_result";
            tool_result["tool_use_id"] = block["id"];

            if (eval_result.success) {
              std::ostringstream oss;
              oss << std::setprecision(15) << eval_result.latency;

              std::string result_str = "Latency: " + oss.str() + " sec\n";

              // Include intermediate variables to help identify dominant terms
              if (!eval_result.variables.empty()) {
                result_str += "\nIntermediate variables:\n";
                for (const auto& [var_name, var_value] :
                     eval_result.variables) {
                  std::ostringstream var_oss;
                  var_oss << std::setprecision(10) << var_value;
                  result_str += var_name + " = " + var_oss.str() + "\n";
                }
              }

              tool_result["content"] = result_str;
            } else {
              tool_result["content"] = "Error: " + eval_result.error_message;
              tool_result["is_error"] = true;
            }
            user_content.push_back(tool_result);
            tool_result_msg["content"] = user_content;

            messages.push_back(tool_result_msg);
          }
        }
      }

      if (!found_tool_use) {
        llvm::report_fatal_error("LLM response did not contain any tool use");
      }
    }
  }

  llvm::report_fatal_error(
      "Tool-use loop exceeded maximum iterations without converging");
}

std::string AgenticTileSizeSelector::buildSystemPrompt(
    llvm::ArrayRef<TileSizeInfo> analyses) {
  std::stringstream ss;
  ss << "You are a compiler optimization expert tasked with selecting optimal "
        "tile sizes for loop tiling through mathematical analysis.\n\n";

  ss << "=== SYMBOLIC COST MODEL ===\n";
  ss << "You have a symbolic cost function that accepts tile size parameters "
        "(s0, s1, ...) "
        "and computes latency through a series of intermediate expressions "
        "(x0, x1, ...).\n\n";
  ss << "The cost function is:\n\n";
  ss << symbolic_cost_function_ << "\n\n";

  ss << "=== MATHEMATICAL ANALYSIS STRATEGY ===\n";
  ss << "CRITICAL: Only call evaluate_cost if you predict it will beat the "
        "current "
        "best result"
        "based on your analysis.\n";
  ss << "BEFORE calling evaluate_cost: State your hypothesis about why you "
        "believe the "
        "assignment should be better than your current best solution (based "
        "only on formula analysis)\n";
  ss << "After seeing results, if prediction was wrong, explain why your "
        "mathematical analysis was incorrect, and learn from your mistakes to "
        "improve future iterations.\n";

  ss << "=== EVALUATION TOOL ===\n";
  ss << "Use the 'evaluate_cost' tool to test tile size assignments:\n";
  ss << "1. Takes an array of tile-size values (s0, s1, ...)\n";
  ss << "2. Evaluates the symbolic cost function with those concrete values\n";
  ss << "3. Returns the final latency in seconds AND all intermediate "
        "variables. These intermediate values are critical in helping find "
        "better solutions, so make sure you use them!\n";
  ss << "4. Use the intermediate variable values to identify which terms "
        "dominate "
        "the latency (critical path)\n";
  ss << "5. Trace back through the formulas to understand how each tile size "
        "affects the largest terms\n\n";

  ss << "Tiling Decision Points:\n";
  for (size_t i = 0; i < analyses.size(); ++i) {
    auto& analysis = const_cast<TileSizeInfo&>(analyses[i]);
    int64_t min_value = analysis.reserve_size_op.getMinValue().getSExtValue();
    int64_t divisibility =
        analysis.reserve_size_op.getDivisibility().getSExtValue();
    ss << "ID " << i << ": min_value=" << min_value
       << ", divisibility=" << divisibility;

    // Add granularity information for this ID
    int64_t granularity = 1;
    for (const auto& loop_info : analysis.associated_loops) {
      auto it = loop_granularities_.find(loop_info.loop);
      if (it != loop_granularities_.end()) {
        granularity = it->second;
        break;  // Use the first loop's granularity (all should be same)
      }
    }
    ss << ", granularity=" << granularity;

    ss << "\n";
    ss << "  Associated loops (total_size): ";
    for (size_t j = 0; j < analysis.associated_loops.size(); ++j) {
      if (j > 0) ss << ", ";
      ss << analysis.associated_loops[j].total_size;
    }
    ss << "\n";
  }

  ss << "\n=== INITIAL HEURISTIC ===\n";
  ss << "A baseline greedy heuristic selects tile sizes as follows:\n";
  ss << "For each tiling decision point ID (independently):\n";
  ss << "1. Start with candidate = max(2, min_value)\n";
  ss << "2. Iterate candidate downward to min_value (stepping by 1)\n";
  ss << "3. Skip any candidate where (candidate % divisibility != 0)\n";
  ss << "4. For each candidate, check if it divides evenly into ALL associated "
        "loop total_sizes\n";
  ss << "   (i.e., for each associated loop, verify: total_size % candidate == "
        "0)\n";
  ss << "5. Return the SMALLEST (first) valid candidate that satisfies all "
        "constraints\n"
     << "YOUR FIRST CALL should evaluate the heuristic-selected tile sizes to "
        "establish baseline latency.\n\n";

  ss << "Your Task:\n";
  ss << "Find the tile sizes that minimize latency. Use evaluate_cost to test "
        "assignments.\n";
  ss << "When satisfied, submit your final answer.\n";

  ss << "Constraints:\n";
  ss << "- Each tile size must be >= min_value\n";
  ss << "- Each tile size must be divisible by its divisibility requirement\n";
  ss << "- CRITICAL: Granularity constraints from parallel regions:\n";
  for (size_t i = 0; i < analyses.size(); ++i) {
    auto& analysis = const_cast<TileSizeInfo&>(analyses[i]);
    int64_t granularity = 1;
    for (const auto& loop_info : analysis.associated_loops) {
      auto it = loop_granularities_.find(loop_info.loop);
      if (it != loop_granularities_.end()) {
        granularity = it->second;
        break;
      }
    }
    ss << "  * ID " << i << ": tile_size must be divisible by " << granularity;
    if (granularity > 1) {
      ss << " (LCM of num_instances from parallel regions in loops)";
    }
    ss << "\n";
  }
  ss << "- When satisfied with your exploration, call submit_final_answer with "
        "the best assignment and your reasoning.\n";

  return ss.str();
}

std::string AgenticTileSizeSelector::buildToolSchemas() {
  json schemas = json::array();

  // evaluate_cost tool
  json evaluate_tool;
  evaluate_tool["name"] = "evaluate_cost";
  evaluate_tool["description"] =
      "Evaluate the symbolic cost function with given tile size assignments. "
      "Returns "
      "final latency and all intermediate variable values.";
  evaluate_tool["input_schema"] = {
      {"type", "object"},
      {"properties",
       {{"tile_sizes",
         {{"type", "array"},
          {"items", {{"type", "integer"}}},
          {"description", "Tile sizes for parameters s0, s1, ... in order"}}},
        {"reasoning", {{"type", "string"}}}}},
      {"required", {"tile_sizes", "reasoning"}}};
  schemas.push_back(evaluate_tool);

  // submit_final_answer tool
  json submit_tool;
  submit_tool["name"] = "submit_final_answer";
  submit_tool["description"] =
      "Submit your final tile-size assignment once satisfied. Provide values "
      "for s0, s1, ... in order.";
  submit_tool["input_schema"] = {
      {"type", "object"},
      {"properties",
       {{"tile_sizes",
         {{"type", "array"},
          {"items", {{"type", "integer"}}},
          {"description", "Tile sizes for parameters s0, s1, ... in order"}}},
        {"explanation", {{"type", "string"}}}}},
      {"required", {"tile_sizes", "explanation"}}};
  schemas.push_back(submit_tool);

  return schemas.dump();
}

AgenticTileSizeSelector::TransformResult
AgenticTileSizeSelector::handleTransformAndEvaluateCost(
    mlir::ModuleOp module, llvm::ArrayRef<TileSizeInfo> analyses,
    const std::vector<std::pair<int64_t, int64_t>>& tile_size_assignments) {
  // Validate tile sizes against granularity constraints
  std::string validation_error;
  if (!validateTileSizeGranularities(analyses, tile_size_assignments,
                                     validation_error)) {
    return {false, 0.0,
            "Granularity constraint violation: " + validation_error};
  }

  // Clone module
  auto cloned_module = llvm::cast<mlir::ModuleOp>(module->clone());

  // Create a mapping from ID to tile size for quick lookup
  std::map<int64_t, int64_t> id_to_tile_size;
  for (const auto& [id, tile_size] : tile_size_assignments) {
    id_to_tile_size[id] = tile_size;
  }

  // Apply tile sizes to the cloned module using applyTileSize helper
  // First, create cloned TileSizeInfo structs that point to cloned ops
  std::vector<mlir::ktdf::TilingReserveSizeOp> cloned_ops;
  cloned_module.walk(
      [&](mlir::ktdf::TilingReserveSizeOp op) { cloned_ops.push_back(op); });

  std::vector<TileSizeInfo> cloned_analyses;
  cloned_analyses.reserve(analyses.size());
  for (size_t i = 0; i < analyses.size() && i < cloned_ops.size(); ++i) {
    TileSizeInfo cloned_info;
    cloned_info.reserve_size_op = cloned_ops[i];
    cloned_info.associated_loops = analyses[i].associated_loops;
    cloned_analyses.push_back(cloned_info);
  }

  // Now apply tile sizes using the same helper as the final pass
  mlir::OpBuilder builder(cloned_module.getContext());
  for (size_t i = 0; i < cloned_analyses.size(); ++i) {
    if (id_to_tile_size.count(i)) {
      int64_t tile_size = id_to_tile_size[i];
      applyTileSize(builder, cloned_analyses[i], tile_size);
    }
  }

  // Print module to string
  std::string module_str;
  llvm::raw_string_ostream ir_stream(module_str);
  cloned_module.print(ir_stream);
  ir_stream.flush();

  // Write to temp file
  llvm::SmallString<256> temp_file;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("tilesize", "mlir", temp_file);
  if (ec) {
    return {false, 0.0, "Failed to create temp file: " + ec.message()};
  }

  std::ofstream f(temp_file.c_str());
  f << module_str;
  f.close();

  // Run cost model subprocess (will return success_status before dumping)
  // We need to check if result succeeded to pass that info to the dumping
  // function
  auto result = runCostModelSubprocess(std::string(temp_file.c_str()),
                                       module_str, tile_size_assignments);

  // If cost model failed, dump IR and terminate immediately
  if (!result.success) {
    dumpDebugIR(module_str, tile_size_assignments, false);
    llvm::errs() << "\n=== COST MODEL FAILED ===\n";
    llvm::errs() << "Tile size assignments:\n";
    for (const auto& [id, tile_size] : tile_size_assignments) {
      llvm::errs() << "  ID " << id << " -> " << tile_size << "\n";
    }
    llvm::errs() << "\nError:\n" << result.error_message << "\n";
    llvm::sys::fs::remove(temp_file);
    llvm::report_fatal_error("Cost model execution failed");
  }

  // Dump successful IR if debug flag is enabled
  if (debug_) {
    dumpDebugIR(module_str, tile_size_assignments, true);
  }

  // Clean up temp file
  llvm::sys::fs::remove(temp_file);

  return result;
}

void AgenticTileSizeSelector::dumpDebugIR(
    const std::string& ir_str,
    const std::vector<std::pair<int64_t, int64_t>>& tile_size_assignments,
    bool success) {
  // Create debug directory structure
  llvm::sys::fs::create_directories("debug/success");
  llvm::sys::fs::create_directories("debug/fail");

  // Build filename from tile sizes: ktdf_1_64.mlir
  std::stringstream filename_ss;
  filename_ss << "ktdf";
  for (const auto& [id, tile_size] : tile_size_assignments) {
    filename_ss << "_" << tile_size;
  }
  filename_ss << ".mlir";
  std::string filename = filename_ss.str();

  // Choose directory based on success
  std::string filepath = success ? "debug/success/" : "debug/fail/";
  filepath += filename;

  // Write IR to file
  std::ofstream debug_file(filepath);
  debug_file << ir_str;
  debug_file.close();
}

AgenticTileSizeSelector::TransformResult
AgenticTileSizeSelector::runCostModelSubprocess(
    const std::string& ir_file, const std::string& ir_str,
    const std::vector<std::pair<int64_t, int64_t>>& tile_size_assignments) {
  // Construct command: cd <cost_model_path> && source samm_env/bin/activate &&
  // python3.12 main.py ...
  std::string cmd = "cd '" + cost_model_path_ +
                    "' && source samm_env/bin/activate && python3.12 main.py "
                    "--mlir-bindings-dir '" +
                    ktdf_bindings_dir_ + "' --input-file '" + ir_file +
                    "' --verbose";

  // Execute command via system() to capture output
  std::string output_file;
  llvm::SmallString<256> temp_out;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("cost_model_out", "txt", temp_out);
  if (ec) {
    return {false, 0.0, "Failed to create output temp file"};
  }
  output_file = std::string(temp_out.c_str());

  // Redirect both stdout and stderr
  std::string full_cmd = cmd + " > " + output_file + " 2>&1";
  int ret_code = system(full_cmd.c_str());

  // Read output
  std::string output_content;
  std::ifstream output_stream(output_file.c_str());
  if (output_stream.is_open()) {
    output_content =
        std::string((std::istreambuf_iterator<char>(output_stream)),
                    std::istreambuf_iterator<char>());
    output_stream.close();
  }

  // Clean up temp file
  llvm::sys::fs::remove(output_file);

  if (ret_code != 0) {
    // Cost model subprocess failed - return error result (don't crash)
    return {false, 0.0,
            "Cost model subprocess exited with code " +
                std::to_string(ret_code) + "\n" + output_content};
  }

  // Parse latency from output - look for last occurrence of "Latency: X sec"
  std::regex latency_regex(R"(Latency:\s*([\d.eE+\-]+)\s*sec)");
  std::smatch match;
  std::string::const_iterator search_start(output_content.cbegin());
  std::string last_match;
  double last_latency = 0.0;

  // Find all matches and use the last one
  while (std::regex_search(search_start, output_content.cend(), match,
                           latency_regex)) {
    last_match = match[1];
    last_latency = std::stod(match[1]);
    search_start = match.suffix().first;
  }

  if (!last_match.empty()) {
    return {true, last_latency, ""};
  }

  // Could not parse latency - return error result
  return {false, 0.0,
          "Could not parse latency from cost model output:\n" + output_content};
}

std::string AgenticTileSizeSelector::makeHttpRequestWithTools(
    const std::string& system_prompt, const json& messages,
    const std::string& tool_schemas) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    llvm::report_fatal_error("Failed to initialize CURL");
  }

  // Build request body
  json request_body;
  request_body["model"] = "aws/claude-opus-4-7";
  request_body["max_tokens"] = 122880;
  request_body["system"] = system_prompt;
  request_body["messages"] = messages;
  request_body["tools"] = json::parse(tool_schemas);

  std::string request_str = request_body.dump();

  // Set CURL options
  curl_easy_setopt(
      curl, CURLOPT_URL,
      "https://ete-litellm.ai-models.vpc-int.res.ibm.com/v1/messages");
  curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_str.c_str());

  // Set headers
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  std::string auth_header = "x-litellm-api-key: " + api_key_;
  headers = curl_slist_append(headers, auth_header.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

  std::string response;
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    std::string error = "CURL request failed: ";
    error += curl_easy_strerror(res);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    llvm::report_fatal_error(llvm::Twine(error));
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  return response;
}

void AgenticTileSizeSelector::computeLoopGranularities(
    llvm::ArrayRef<TileSizeInfo> analyses) {
  // For each loop in the analyses, compute the LCM of num_instances
  // of all parallel regions in its body
  for (const auto& analysis : analyses) {
    for (const auto& loop_info : analysis.associated_loops) {
      auto loop = loop_info.loop;

      // Find all parallel regions in this loop's body
      int64_t granularity = 1;
      loop.walk([&](mlir::ktdf::ParallelOp parallel_op) {
        int64_t num_instances = parallel_op.getNumInstances();
        // Compute LCM(granularity, num_instances)
        granularity = (granularity / std::gcd(granularity, num_instances)) *
                      num_instances;
      });

      loop_granularities_[loop] = granularity;
    }
  }
}

bool AgenticTileSizeSelector::validateTileSizeGranularities(
    llvm::ArrayRef<TileSizeInfo> analyses,
    const std::vector<std::pair<int64_t, int64_t>>& tile_size_assignments,
    std::string& error_message) {
  for (size_t i = 0; i < analyses.size(); ++i) {
    auto& analysis = const_cast<TileSizeInfo&>(analyses[i]);

    // Find the tile size for this analysis
    int64_t tile_size = -1;
    for (const auto& [id, ts] : tile_size_assignments) {
      if (id == (int64_t)i) {
        tile_size = ts;
        break;
      }
    }

    if (tile_size < 0) continue;  // Not assigned yet

    // Check granularity constraint for each associated loop
    for (const auto& loop_info : analysis.associated_loops) {
      auto it = loop_granularities_.find(loop_info.loop);
      if (it == loop_granularities_.end()) continue;

      int64_t granularity = it->second;
      if (granularity > 1 && tile_size % granularity != 0) {
        error_message = "Tile size " + std::to_string(tile_size) +
                        " for loop ID " + std::to_string(i) +
                        " must be divisible by " + std::to_string(granularity) +
                        " (LCM of parallel region num_instances)";
        return false;
      }
    }
  }

  return true;
}

}  // namespace scheduler
