//===-- AnthropicAgentClient.cpp ---------*- c++ -*-===//
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

#include "dataflow-scheduler/Utils/AnthropicAgentClient.h"

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#include "nlohmann/json.hpp"
#pragma clang diagnostic pop

#include <curl/curl.h>
#include <sstream>

using json = nlohmann::json;

namespace scheduler {

// Forward declaration for libcurl callback
static size_t WriteCallback(void* contents, size_t size, size_t nmemb,
                            std::string* s) {
  s->append((char*)contents, size * nmemb);
  return size * nmemb;
}

AnthropicAgentClient::AnthropicAgentClient(const std::string& api_key)
    : api_key_(api_key) {}

AnthropicAgentClient::~AnthropicAgentClient() = default;

int64_t AnthropicAgentClient::selectTileSize(
    mlir::ModuleOp module,
    TileSizeInfo& tile_size_info) {
  std::string prompt = buildPrompt(module, tile_size_info);
  std::string response = makeHttpRequest(prompt);
  int64_t tile_size = parseJsonResponse(response);

  llvm::errs() << "[Agent] Selected tile_size: " << tile_size << "\n";
  return tile_size;
}

std::string AnthropicAgentClient::buildPrompt(
    mlir::ModuleOp module,
    TileSizeInfo& tile_size_info) {
  int64_t min_value =
      tile_size_info.reserve_size_op.getMinValue().getSExtValue();
  int64_t divisibility =
      tile_size_info.reserve_size_op.getDivisibility().getSExtValue();

  std::string prompt;
  prompt += buildTaskDefinition();
  prompt += "\n\n";
  prompt += buildConstraintsSection(tile_size_info, min_value, divisibility);
  prompt += "\n\n";
  prompt += buildContextSection(module, tile_size_info);
  prompt += "\n\n";
  prompt += buildHeuristicBaselineSection();
  prompt += "\n\n";
  prompt += buildOutputFormatSection(min_value);

  return prompt;
}

std::string AnthropicAgentClient::buildTaskDefinition() {
  return R"(You are a compiler optimization expert specializing in dataflow scheduling.

Your task: Given a tile size selection problem in a dataflow scheduler,
recommend an optimal tile size.

Context: We are scheduling computations for dataflow hardware accelerators.
Tile sizes determine how much work is executed in each hardware iteration.
Larger tiles improve hardware utilization but may increase memory pressure.
Smaller tiles reduce memory usage but increase control overhead.

Your goal is to select a tile size that optimizes performance/efficiency
tradeoffs given the constraints.)";
}

std::string AnthropicAgentClient::buildConstraintsSection(
    TileSizeInfo& tile_size_info,
    int64_t min_value,
    int64_t divisibility) {
  std::ostringstream ss;
  ss << "CONSTRAINTS (your response MUST satisfy ALL of these):\n\n";
  ss << "1. Minimum value constraint:\n";
  ss << "   selected_tile_size >= " << min_value << "\n\n";
  ss << "2. Divisibility constraint:\n";
  ss << "   selected_tile_size % " << divisibility << " == 0\n";
  if (divisibility > 1) {
    ss << "   (must be a multiple of " << divisibility << ")\n";
  } else {
    ss << "   (no divisibility constraint)\n";
  }
  ss << "\n3. Loop eviction constraint (critical):\n";
  ss << "   For each loop being tiled, the total loop size must be evenly "
        "divisible\n";
  ss << "   by the tile size.\n\n";
  ss << "   LOOPS:\n";
  for (size_t i = 0; i < tile_size_info.associated_loops.size(); ++i) {
    ss << "     Loop " << i
       << ": trip_count = " << tile_size_info.associated_loops[i].total_size
       << "\n";
  }
  ss << "\n   REQUIREMENT: For ALL loops above:\n";
  ss << "     (trip_count % selected_tile_size) == 0\n";

  return ss.str();
}

std::string AnthropicAgentClient::buildContextSection(
    mlir::ModuleOp module,
    TileSizeInfo& tile_size_info) {
  int64_t min_value =
      tile_size_info.reserve_size_op.getMinValue().getSExtValue();
  int64_t divisibility =
      tile_size_info.reserve_size_op.getDivisibility().getSExtValue();

  std::ostringstream ss;
  ss << "PROBLEM INSTANCE:\n\n";
  ss << "Constraints:\n";
  ss << "  - Minimum tile size: " << min_value << "\n";
  ss << "  - Divisibility constraint: " << divisibility << "\n\n";
  ss << "Associated Loops:\n";
  for (size_t i = 0; i < tile_size_info.associated_loops.size(); ++i) {
    int64_t trip_count = tile_size_info.associated_loops[i].total_size;
    ss << "  Loop " << i << ":\n";
    ss << "    - Trip count: " << trip_count << "\n";
    ss << "    - Valid tile sizes: {";

    // Compute valid divisors
    std::vector<int64_t> valid_sizes;
    for (int64_t candidate = min_value; candidate <= trip_count; ++candidate) {
      if (trip_count % candidate == 0 &&
          (divisibility <= 1 || candidate % divisibility == 0)) {
        valid_sizes.push_back(candidate);
      }
    }

    for (size_t j = 0; j < valid_sizes.size(); ++j) {
      ss << valid_sizes[j];
      if (j < valid_sizes.size() - 1) ss << ", ";
    }
    ss << "}\n";
  }

  ss << "\nSCHEDULE IR (partial, first 50 lines):\n```mlir\n";
  std::string ir_str;
  {
    llvm::raw_string_ostream ss_ir(ir_str);
    module.print(ss_ir);
  }

  std::istringstream ir_stream(ir_str);
  std::string line;
  int line_count = 0;
  while (std::getline(ir_stream, line) && line_count < 50) {
    ss << line << "\n";
    line_count++;
  }

  if (line_count >= 50) {
    ss << "... (IR truncated for brevity) ...\n";
  }
  ss << "```\n";

  return ss.str();
}

std::string AnthropicAgentClient::buildHeuristicBaselineSection() {
  return R"(HEURISTIC BASELINE (for reference):

The existing scheduling compiler uses this heuristic:
1. Start from max_candidate = 2 (or min_value if higher)
2. Search downward from max_candidate to min_value
3. Accept the first candidate where:
   - candidate % divisibility == 0 (if divisibility > 1)
   - For ALL loops: trip_count % candidate == 0
4. Worst case: tile_size = 1 (always valid)

This heuristic prioritizes SIMPLICITY over optimization.

For this exercise, please reproduce this heuristic behavior exactly.)";
}

std::string AnthropicAgentClient::buildOutputFormatSection(int64_t min_value) {
  std::ostringstream ss;
  ss << "OUTPUT FORMAT (respond ONLY with valid JSON):\n\n";
  ss << "{\n";
  ss << "  \"selected_tile_size\": <integer>\n";
  ss << "}\n\n";
  ss << "IMPORTANT:\n";
  ss << "- selected_tile_size must be >= " << min_value << "\n";
  ss << "- Respond with ONLY the JSON object, no other text\n";
  ss << "- Your suggestion WILL BE VALIDATED against the constraints above\n";
  ss << "- Invalid suggestions will cause the compiler to TERMINATE\n";

  return ss.str();
}

std::string AnthropicAgentClient::makeHttpRequest(const std::string& prompt) {
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

  curl_easy_setopt(curl, CURLOPT_URL,
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

int64_t AnthropicAgentClient::parseJsonResponse(const std::string& response) {
  json response_obj = json::parse(response);

  // Check for API errors first
  if (response_obj.contains("type") && response_obj["type"] == "error") {
    std::string error_msg = "API Error: ";
    if (response_obj.contains("error")) {
      auto& error = response_obj["error"];
      if (error.contains("type")) {
        error_msg += error["type"].get<std::string>();
        error_msg += " - ";
      }
      if (error.contains("message")) {
        error_msg += error["message"].get<std::string>();
      }
    }
    llvm::report_fatal_error(llvm::StringRef(error_msg));
  }

  // Navigate to the response content
  if (response_obj.contains("content") && response_obj["content"].is_array() &&
      response_obj["content"].size() > 0) {
    std::string text = response_obj["content"][0]["text"];

    // Try to find JSON in the text - it might be wrapped in markdown or text
    size_t json_start = text.find('{');
    if (json_start != std::string::npos) {
      size_t json_end = text.rfind('}');
      if (json_end != std::string::npos && json_end > json_start) {
        std::string json_str = text.substr(json_start, json_end - json_start + 1);

        json result = json::parse(json_str);
        if (result.contains("selected_tile_size")) {
          int64_t tile_size = result["selected_tile_size"].get<int64_t>();
          return tile_size;
        }
      }
    }
  }

  llvm::report_fatal_error(llvm::StringRef("Response missing selected_tile_size field"));
  return 0;
}

}  // namespace scheduler
