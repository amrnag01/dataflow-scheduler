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
    const std::string& cost_model_path,
    const std::string& learnings_file_path, bool debug)
    : api_key_(api_key),
      ktdf_bindings_dir_(ktdf_bindings_dir),
      cost_model_path_(cost_model_path),
      learnings_file_path_(learnings_file_path),
      debug_(debug) {
  // Load learnings file
  std::ifstream learnings_file(learnings_file_path);
  if (!learnings_file.is_open()) {
    llvm::report_fatal_error(
        llvm::Twine("Failed to open learnings file: ") + learnings_file_path);
  }
  learnings_content_ =
      std::string((std::istreambuf_iterator<char>(learnings_file)),
                  std::istreambuf_iterator<char>());
  learnings_file.close();
}

AgenticTileSizeSelector::~AgenticTileSizeSelector() = default;

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

  std::string system_prompt = buildSystemPrompt(analyses);
  std::string tool_schemas = buildToolSchemas();

  std::vector<json> messages;
  json user_msg;
  user_msg["role"] = "user";
  user_msg["content"] =
      "You are optimizing tile sizes for a compiler. Use the "
      "transform_and_evaluate_cost "
      "tool to explore different tile-size assignments, and when satisfied, "
      "call "
      "submit_final_answer with your best choice and reasoning.";
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
            std::vector<int64_t> result(analyses.size());

            for (const auto& ts : input["tile_sizes"]) {
              int64_t id = ts["id"];
              int64_t tile_size = ts["tile_size"];

              if (id < 0 || id >= (int64_t)analyses.size()) {
                llvm::report_fatal_error(
                    llvm::Twine("Final answer references invalid id: ") +
                    std::to_string(id));
              }
              result[id] = tile_size;
            }

            // Validate each tile size
            for (size_t i = 0; i < analyses.size(); ++i) {
              auto& analysis = const_cast<TileSizeInfo&>(analyses[i]);
              int64_t tile_size = result[i];
              int64_t min_value =
                  analysis.reserve_size_op.getMinValue().getSExtValue();
              int64_t divisibility =
                  analysis.reserve_size_op.getDivisibility().getSExtValue();

              if (tile_size < min_value) {
                llvm::report_fatal_error(
                    llvm::Twine("Final tile size for op ") + std::to_string(i) +
                    " is " + std::to_string(tile_size) + " but minimum is " +
                    std::to_string(min_value));
              }

              if (tile_size % divisibility != 0) {
                llvm::report_fatal_error(
                    llvm::Twine("Final tile size for op ") + std::to_string(i) +
                    " is " + std::to_string(tile_size) +
                    " but must be divisible by " +
                    std::to_string(divisibility));
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

            // Handle learnings_update
            if (input.contains("learnings_update")) {
              auto learnings_array = input["learnings_update"];
              if (learnings_array.is_array() && !learnings_array.empty()) {
                llvm::errs() << "[Agent] New Learnings:\n";
                std::vector<std::string> new_learnings;
                for (const auto& learning : learnings_array) {
                  if (learning.is_string()) {
                    std::string learning_str = learning.get<std::string>();
                    llvm::errs() << learning_str << "\n";
                    new_learnings.push_back(learning_str);
                  }
                }
                // Append validated learnings to file
                appendLearningstToFile(new_learnings);
              } else {
                llvm::errs() << "[Agent] No new learnings (prior learnings were sufficient)\n";
              }
            }

            return result;
          } else if (tool_name == "transform_and_evaluate_cost") {
            // Execute tool
            json input = block["input"];
            std::vector<std::pair<int64_t, int64_t>> tile_size_assignments;

            for (const auto& ts : input["tile_sizes"]) {
              int64_t id = ts["id"];
              int64_t tile_size = ts["tile_size"];
              tile_size_assignments.push_back({id, tile_size});
            }

            std::string reasoning = input["reasoning"].get<std::string>();

            // Print what we're trying before executing
            llvm::errs() << "[Agent] Trying: ";
            for (size_t i = 0; i < tile_size_assignments.size(); ++i) {
              llvm::errs() << (i > 0 ? ", " : "")
                           << tile_size_assignments[i].second;
            }
            llvm::errs() << "\n";

            auto result = handleTransformAndEvaluateCost(module, analyses,
                                                         tile_size_assignments);

            llvm::errs() << "Reasoning: " << reasoning << "\n";
            if (result.success) {
              std::ostringstream latency_str;
              latency_str << std::setprecision(15) << result.latency;
              llvm::errs() << "Latency: " << latency_str.str() << " sec\n";
            } else {
              llvm::errs() << "Error: " << result.error_message << "\n";
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
            if (result.success) {
              // Format with full precision
              std::ostringstream oss;
              oss << std::setprecision(15) << result.latency;
              tool_result["content"] = "Latency: " + oss.str() + " sec";
            } else {
              tool_result["content"] = "Error: " + result.error_message;
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
        "tile sizes for loop tiling.\n\n";

  ss << "=== LEARNINGS FROM PREVIOUS RUNS ===\n\n";
  ss << learnings_content_ << "\n\n";

  ss << "=== TILING DECISION POINTS ===\n";
  for (size_t i = 0; i < analyses.size(); ++i) {
    auto& analysis = const_cast<TileSizeInfo&>(analyses[i]);

    // Add granularity information for this ID
    int64_t granularity = 1;
    for (const auto& loop_info : analysis.associated_loops) {
      auto it = loop_granularities_.find(loop_info.loop);
      if (it != loop_granularities_.end()) {
        granularity = it->second;
        break;
      }
    }
    ss << "ID " << i << ": granularity=" << granularity;

    ss << "\n";
    ss << "  Associated loops (total_size): ";
    for (size_t j = 0; j < analysis.associated_loops.size(); ++j) {
      if (j > 0) ss << ", ";
      ss << analysis.associated_loops[j].total_size;
    }
    ss << "\n";
  }

  ss << "\n=== YOUR TASK ===\n";
  ss << "1. Use the learnings above to guide your tile-size exploration\n";
  ss << "2. Call transform_and_evaluate_cost to explore tile-size candidates\n";
  ss << "3. When satisfied, call submit_final_answer with your best choice and reasoning\n";
  ss << "4. If previous learnings were sufficient to find the optimal solution, set learnings_update to empty array\n";
  ss << "5. When adding new learnings, use EXACTLY this template (do not deviate):\n\n";
  ss << "### Learning: [Short Title]\n\n";
  ss << "**Key Insight**: [One-sentence general rule discovered]\n\n";
  ss << "**Pattern**: [Describe the pattern: when does this apply and what should be done]\n\n";
  ss << "**Example**:\n";
  ss << "```\n";
  ss << "[Loop structure or code snippet showing the pattern]\n\n";
  ss << "Analysis:\n";
  ss << "  - [Key observation 1]\n";
  ss << "  - [Key observation 2]\n";
  ss << "  - Decision: [Resulting tile-size choice and why]\n";
  ss << "```\n\n";
  ss << "IMPORTANT: Every learning must include all four sections (Key Insight, Pattern, Example with Analysis) in exactly this order. Start each new learning on a new line.\n";

  return ss.str();
}

std::string AgenticTileSizeSelector::buildToolSchemas() {
  json schemas = json::array();

  // transform_and_evaluate_cost tool
  json transform_tool;
  transform_tool["name"] = "transform_and_evaluate_cost";
  transform_tool["description"] =
      "Apply tile-size assignments to the IR and measure latency via SAMM cost "
      "model";
  transform_tool["input_schema"] = {
      {"type", "object"},
      {"properties",
       {{"tile_sizes",
         {{"type", "array"},
          {"items",
           {{"type", "object"},
            {"properties",
             {{"id", {{"type", "integer"}}},
              {"tile_size", {{"type", "integer"}}}}},
            {"required", {"id", "tile_size"}}}}}},
        {"reasoning", {{"type", "string"}}}}},
      {"required", {"tile_sizes", "reasoning"}}};
  schemas.push_back(transform_tool);

  // submit_final_answer tool
  json submit_tool;
  submit_tool["name"] = "submit_final_answer";
  submit_tool["description"] =
      "Submit your final tile-size assignment once satisfied. Include new learnings if discovered, or empty array if prior learnings were sufficient.";
  submit_tool["input_schema"] = {
      {"type", "object"},
      {"properties",
       {{"tile_sizes",
         {{"type", "array"},
          {"items",
           {{"type", "object"},
            {"properties",
             {{"id", {{"type", "integer"}}},
              {"tile_size", {{"type", "integer"}}}}},
            {"required", {"id", "tile_size"}}}}}},
        {"explanation", {{"type", "string"}}},
        {"learnings_update",
         {{"type", "array"},
          {"items", {{"type", "string"}}},
          {"description",
           "Array of new learnings discovered (as markdown strings in Learning "
           "template format), or empty array if no new learnings found"}}}}},
      {"required", {"tile_sizes", "explanation", "learnings_update"}}};
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

void AgenticTileSizeSelector::appendLearningstToFile(
    const std::vector<std::string>& learnings) {
  if (learnings.empty()) {
    return;
  }

  // Validate that each learning matches the template structure
  for (const auto& learning : learnings) {
    // Check for required sections: ### Learning:, **Key Insight**:, **Pattern**:, **Example**:, Analysis:
    if (learning.find("### Learning:") == std::string::npos) {
      llvm::errs() << "[Agent] Warning: Learning missing '### Learning:' header\n";
      continue;
    }
    if (learning.find("**Key Insight**:") == std::string::npos) {
      llvm::errs() << "[Agent] Warning: Learning missing '**Key Insight**:' section\n";
      continue;
    }
    if (learning.find("**Pattern**:") == std::string::npos) {
      llvm::errs() << "[Agent] Warning: Learning missing '**Pattern**:' section\n";
      continue;
    }
    if (learning.find("**Example**:") == std::string::npos) {
      llvm::errs() << "[Agent] Warning: Learning missing '**Example**:' section\n";
      continue;
    }
    if (learning.find("Analysis:") == std::string::npos) {
      llvm::errs() << "[Agent] Warning: Learning missing 'Analysis:' section\n";
      continue;
    }

    // Append to file
    std::ofstream learnings_file(learnings_file_path_, std::ios::app);
    if (!learnings_file.is_open()) {
      llvm::errs() << "[Agent] Error: Failed to open learnings file for appending: "
                   << learnings_file_path_ << "\n";
      continue;
    }

    learnings_file << "\n---\n\n" << learning << "\n";
    learnings_file.close();

    llvm::errs() << "[Agent] Successfully appended new learning to " << learnings_file_path_
                 << "\n";
  }
}

}  // namespace scheduler
