//===-- KTDFOptimizationAgent.cpp ---------*- c++ -*-===//
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

#include "dataflow-scheduler/Utils/KTDFOptimizationAgent.h"

#include <iomanip>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Parser/Parser.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#include "nlohmann/json.hpp"
#pragma clang diagnostic pop

#include <curl/curl.h>

#include <sstream>

using json = nlohmann::json;

namespace scheduler {

static size_t WriteCallback(void* contents, size_t size, size_t nmemb,
                            std::string* s) {
  s->append((char*)contents, size * nmemb);
  return size * nmemb;
}

KTDFOptimizationAgent::KTDFOptimizationAgent(
    const std::string& api_key, const std::string& ktdf_bindings_dir,
    const std::string& cost_model_path)
    : api_key_(api_key),
      ktdf_bindings_dir_(ktdf_bindings_dir),
      cost_model_path_(cost_model_path) {}

KTDFOptimizationAgent::~KTDFOptimizationAgent() = default;

mlir::ModuleOp KTDFOptimizationAgent::optimizeKTDF(mlir::ModuleOp module) {
  llvm::errs() << "[KTDFOptimizationAgent] Starting optimization loop\n";

  // Get IR string
  std::string ir_str;
  {
    llvm::raw_string_ostream ss(ir_str);
    module.print(ss);
  }

  std::string system_prompt = buildSystemPrompt();
  std::string tool_schemas = buildToolSchemas();

  // Build initial user message with IR
  std::ostringstream initial_content;
  initial_content
      << "KTDF IR to optimize:\n\n```mlir\n"
      << ir_str << "\n```\n\n"
      << "TASK:\n"
      << "1. Call the evaluate_cost tool once to measure baseline latency\n"
      << "2. Then call submit_final_answer\n"
      << "3. In your response, include the optimized IR in a ```mlir code "
         "block\n\n"
      << "For this test, return the IR unchanged. Include it in your response "
         "as:\n"
      << "```mlir\n[the complete IR here]\n```\n";

  json messages = json::array();
  json user_msg;
  user_msg["role"] = "user";
  user_msg["content"] = initial_content.str();
  messages.push_back(user_msg);

  // Tool-use loop
  int iteration = 0;
  const int max_iterations = 20;

  while (iteration < max_iterations) {
    iteration++;
    llvm::errs() << "[KTDFOptimizationAgent] Iteration " << iteration << "\n";

    // Build request
    json request_body;
    request_body["model"] = "aws/claude-opus-5";
    request_body["max_tokens"] = 16000;
    request_body["system"] = system_prompt;
    request_body["tools"] = json::parse(tool_schemas);
    request_body["messages"] = messages;

    std::string request_str = request_body.dump();

    // Make HTTP request
    CURL* curl = curl_easy_init();
    if (!curl) {
      llvm::report_fatal_error("Failed to initialize libcurl");
    }

    std::string response_str;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "accept: application/json");

    std::string auth_header = "x-litellm-api-key: " + api_key_;
    headers = curl_slist_append(headers, auth_header.c_str());

    curl_easy_setopt(
        curl, CURLOPT_URL,
        "https://ete-litellm.ai-models.vpc-int.res.ibm.com/v1/messages");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_str.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_str);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      std::string error_msg =
          std::string("Curl request failed: ") + curl_easy_strerror(res);
      curl_easy_cleanup(curl);
      curl_slist_free_all(headers);
      llvm::report_fatal_error(llvm::StringRef(error_msg));
    }

    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);

    // Parse response
    json response_obj = json::parse(response_str);

    if (response_obj.contains("type") && response_obj["type"] == "error") {
      std::string error_msg = "API Error: ";
      if (response_obj.contains("error")) {
        auto& error = response_obj["error"];
        if (error.contains("message")) {
          error_msg += error["error"]["message"].get<std::string>();
        }
      }
      llvm::report_fatal_error(llvm::StringRef(error_msg));
    }

    // Extract assistant message
    json assistant_msg;
    assistant_msg["role"] = "assistant";
    assistant_msg["content"] = json::array();

    bool handled_tool = false;

    std::string full_response_text;

    llvm::errs() << "[KTDFOptimizationAgent] Full response JSON:\n"
                 << response_obj.dump(2) << "\n";

    if (response_obj.contains("content") &&
        response_obj["content"].is_array()) {
      llvm::errs() << "[KTDFOptimizationAgent] Content blocks count: "
                   << response_obj["content"].size() << "\n";

      for (const auto& block : response_obj["content"]) {
        if (block.contains("type")) {
          std::string block_type = block["type"];
          llvm::errs() << "[KTDFOptimizationAgent] Block type: " << block_type
                       << "\n";

          if (block_type == "text") {
            std::string text = block["text"];
            llvm::errs() << "[KTDFOptimizationAgent] Text block: " << text
                         << "\n";
            full_response_text += text + "\n";

            json text_content;
            text_content["type"] = "text";
            text_content["text"] = text;
            assistant_msg["content"].push_back(text_content);
          } else if (block_type == "thinking") {
            std::string thinking = block["thinking"];
            llvm::errs() << "[KTDFOptimizationAgent] Thinking block (len: "
                         << thinking.size() << "): " << thinking << "\n";
            // Don't add thinking to full_response_text
          } else if (block_type == "tool_use") {
            std::string tool_name = block["name"];
            llvm::errs() << "[KTDFOptimizationAgent] Tool use: " << tool_name
                         << "\n";

            json tool_use_content;
            tool_use_content["type"] = "tool_use";
            tool_use_content["id"] = block["id"];
            tool_use_content["name"] = block["name"];
            tool_use_content["input"] = block["input"];
            assistant_msg["content"].push_back(tool_use_content);

            // Handle tool execution
            if (tool_name == "submit_final_answer") {
              // Try to get IR from tool input field first
              json input = block["input"];
              llvm::errs()
                  << "[KTDFOptimizationAgent] submit_final_answer input: "
                  << input.dump() << "\n";

              std::string optimized_ir;

              if (input.contains("optimized_ir") &&
                  input["optimized_ir"].is_string()) {
                optimized_ir = input["optimized_ir"].get<std::string>();
                llvm::errs() << "[KTDFOptimizationAgent] Found optimized_ir in "
                                "tool input (size: "
                             << optimized_ir.size() << " bytes)\n";
              }

              // If not in tool input, look in response text
              if (optimized_ir.empty()) {
                llvm::errs() << "[KTDFOptimizationAgent] optimized_ir not in "
                                "input, looking in response text...\n";
                size_t ir_start = full_response_text.find("```mlir");
                if (ir_start != std::string::npos) {
                  ir_start += 7;  // Skip "```mlir"
                  size_t ir_end = full_response_text.find("```", ir_start);
                  if (ir_end != std::string::npos) {
                    optimized_ir =
                        full_response_text.substr(ir_start, ir_end - ir_start);

                    // Skip leading whitespace/newlines
                    size_t first = optimized_ir.find_first_not_of(" \t\n\r");
                    if (first != std::string::npos) {
                      optimized_ir = optimized_ir.substr(first);
                    }

                    // Skip trailing whitespace
                    size_t last = optimized_ir.find_last_not_of(" \t\n\r");
                    if (last != std::string::npos) {
                      optimized_ir = optimized_ir.substr(0, last + 1);
                    }
                  }
                }
              }

              if (!optimized_ir.empty()) {
                // Write optimized IR to temp file
                llvm::SmallString<128> temp_path;
                std::error_code ec = llvm::sys::fs::createTemporaryFile(
                    "ktdf-optimized", ".mlir", temp_path);
                if (ec) {
                  llvm::report_fatal_error(
                      "[KTDFOptimizationAgent] FATAL: Failed to create temp file");
                }

                llvm::raw_fd_ostream temp_file(temp_path, ec);
                if (ec) {
                  llvm::report_fatal_error(
                      "[KTDFOptimizationAgent] FATAL: Failed to open temp file");
                }

                temp_file << optimized_ir;
                temp_file.close();

                llvm::errs() << "[KTDFOptimizationAgent] Wrote optimized IR to "
                             << temp_path << "\n";

                // Store temp file path for pass to read
                // We return the original module for now; the pass will handle the
                // file-based replacement
                optimized_ir_path_ = temp_path.str().str();
                return module;
              }

              llvm::errs() << "[KTDFOptimizationAgent] No optimized IR found, "
                              "using original module\n";
              // Fallback: return original module
              return module;
            } else if (tool_name == "evaluate_cost" && !handled_tool) {
              handled_tool = true;

              json input = block["input"];
              std::string reasoning = input["reasoning"].get<std::string>();

              llvm::errs() << "[KTDFOptimizationAgent] Evaluating cost\n";
              llvm::errs() << "Reasoning: " << reasoning << "\n";

              auto eval_result = evaluateCost(module);

              if (eval_result.success) {
                llvm::errs() << "Latency: " << eval_result.latency << " sec\n";
              } else {
                llvm::errs() << "Error: " << eval_result.error_message << "\n";
              }

              // Add assistant message to history
              messages.push_back(assistant_msg);

              // Add tool result
              json user_content = json::array();
              json tool_result;
              tool_result["type"] = "tool_result";
              tool_result["tool_use_id"] = block["id"];

              if (eval_result.success) {
                std::ostringstream oss;
                oss << std::setprecision(15) << eval_result.latency;
                tool_result["content"] = "Latency: " + oss.str() + " sec";
              } else {
                tool_result["content"] = "Error: " + eval_result.error_message;
                tool_result["is_error"] = true;
              }
              user_content.push_back(tool_result);

              json user_response;
              user_response["role"] = "user";
              user_response["content"] = user_content;
              messages.push_back(user_response);

              // Add explicit reminder to call submit_final_answer with all
              // fields
              json reminder;
              reminder["role"] = "user";
              std::ostringstream reminder_content;
              reminder_content
                  << "Now call submit_final_answer with:\n"
                  << "1. optimized_ir: the complete MLIR module (for this "
                     "test, return the IR unchanged)\n"
                  << "2. explanation: what optimizations were attempted\n"
                  << "Remember: Both fields are REQUIRED.\n";
              reminder["content"] = reminder_content.str();
              messages.push_back(reminder);

              // Continue to next iteration of while loop
              break;
            }
          }
        }
      }
    }

    // If we didn't handle a tool, add assistant message and check if we should
    // continue
    if (!handled_tool) {
      messages.push_back(assistant_msg);

      // Check for stop_reason
      if (response_obj.contains("stop_reason") &&
          response_obj["stop_reason"] == "end_turn") {
        llvm::errs()
            << "[KTDFOptimizationAgent] Agent ended turn without tool use\n";
        return module;
      }
    }
  }

  llvm::errs() << "[KTDFOptimizationAgent] Max iterations reached\n";
  return module;
}

std::string KTDFOptimizationAgent::makeHttpRequest(const std::string& prompt) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    llvm::report_fatal_error("Failed to initialize libcurl");
  }

  json request_body;
  request_body["model"] = "aws/claude-opus-4-7";
  request_body["max_tokens"] = 1024;
  request_body["messages"] = json::array();
  request_body["messages"][0]["role"] = "user";
  request_body["messages"][0]["content"] = prompt;

  std::string request_str = request_body.dump();
  std::string response_str;

  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "accept: application/json");

  std::string auth_header = "x-litellm-api-key: " + api_key_;
  headers = curl_slist_append(headers, auth_header.c_str());

  curl_easy_setopt(
      curl, CURLOPT_URL,
      "https://ete-litellm.ai-models.vpc-int.res.ibm.com/v1/messages");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_str.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_str);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    std::string error_msg =
        std::string("Curl request failed: ") + curl_easy_strerror(res);
    curl_easy_cleanup(curl);
    curl_slist_free_all(headers);
    llvm::report_fatal_error(llvm::StringRef(error_msg));
  }

  curl_easy_cleanup(curl);
  curl_slist_free_all(headers);

  return response_str;
}

std::string KTDFOptimizationAgent::buildSystemPrompt() {
  std::ostringstream ss;
  ss << "You are a compiler optimization expert specializing in KTDF IR "
        "optimization.\n\n";
  ss << "CRITICAL - REQUIRED TOOL BEHAVIOR:\n";
  ss << "When you call the submit_final_answer tool, you MUST include BOTH of "
        "these fields in the tool input:\n";
  ss << "1. explanation: A brief description of what you did (can be empty "
        "string if no changes)\n";
  ss << "2. optimized_ir: The COMPLETE MLIR module as a string. This field is "
        "MANDATORY. The compiler will fail if it is missing or empty.\n\n";
  ss << "Your task: Optimize the given KTDF IR to minimize latency while "
        "maintaining functional correctness.\n";
  return ss.str();
}

std::string KTDFOptimizationAgent::buildToolSchemas() {
  json schemas = json::array();

  // evaluate_cost tool
  json evaluate_tool;
  evaluate_tool["name"] = "evaluate_cost";
  evaluate_tool["description"] =
      "Evaluate latency of current IR using SAMM cost model";
  json eval_schema;
  eval_schema["type"] = "object";
  eval_schema["properties"]["reasoning"] = {
      {"type", "string"},
      {"description", "Why you are evaluating this configuration"}};
  eval_schema["required"] = json::array({"reasoning"});
  evaluate_tool["input_schema"] = eval_schema;
  schemas.push_back(evaluate_tool);

  // submit_final_answer tool
  json submit_tool;
  submit_tool["name"] = "submit_final_answer";
  submit_tool["description"] =
      "Submit final optimized IR when satisfied. MUST include optimized_ir "
      "with full MLIR module text.";
  json submit_schema;
  submit_schema["type"] = "object";
  submit_schema["properties"]["optimized_ir"] = {
      {"type", "string"},
      {"description",
       "The optimized KTDF IR as a complete MLIR module - this is REQUIRED"}};
  submit_schema["properties"]["explanation"] = {
      {"type", "string"},
      {"description",
       "Explanation of optimizations applied - this is REQUIRED"}};
  submit_schema["required"] = json::array({"optimized_ir", "explanation"});
  submit_tool["input_schema"] = submit_schema;
  schemas.push_back(submit_tool);

  return schemas.dump();
}

KTDFOptimizationAgent::CostEvaluation KTDFOptimizationAgent::evaluateCost(
    mlir::ModuleOp module) {
  llvm::errs() << "[KTDFOptimizationAgent] Placeholder: evaluating cost\n";
  // Placeholder: for now just return a dummy latency value
  return {true, 1.0, {}, ""};
}

}  // namespace scheduler
