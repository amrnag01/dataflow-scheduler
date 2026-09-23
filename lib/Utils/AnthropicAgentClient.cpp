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

#include "llvm/Support/raw_ostream.h"

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

AnthropicAgentClient::AnthropicAgentClient(const std::string& api_key)
    : api_key_(api_key) {}

AnthropicAgentClient::~AnthropicAgentClient() = default;

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

  if (response_obj.contains("content") && response_obj["content"].is_array() &&
      response_obj["content"].size() > 0) {
    std::string text = response_obj["content"][0]["text"];

    size_t json_start = text.find('{');
    if (json_start != std::string::npos) {
      size_t json_end = text.rfind('}');
      if (json_end != std::string::npos && json_end > json_start) {
        std::string json_str = text.substr(json_start, json_end - json_start + 1);

        json result = json::parse(json_str);
        if (result.contains("result")) {
          int64_t value = result["result"].get<int64_t>();
          return value;
        }
      }
    }
  }

  llvm::report_fatal_error(llvm::StringRef("Response missing result field"));
  return 0;
}

int64_t AnthropicAgentClient::optimizeKTDF(mlir::ModuleOp module) {
  std::string prompt =
      "You are a compiler optimization expert. Please analyze the following IR and return the number 2.\n\n"
      "IR:\n```mlir\n";

  std::string ir_str;
  {
    llvm::raw_string_ostream ss(ir_str);
    module.print(ss);
  }

  std::istringstream ir_stream(ir_str);
  std::string line;
  int line_count = 0;
  while (std::getline(ir_stream, line) && line_count < 50) {
    prompt += line + "\n";
    line_count++;
  }

  if (line_count >= 50) {
    prompt += "... (IR truncated) ...\n";
  }

  prompt += "```\n\n";
  prompt += "Respond ONLY with valid JSON:\n";
  prompt += "{\n";
  prompt += "  \"result\": 2\n";
  prompt += "}\n";

  std::string response = makeHttpRequest(prompt);

  json response_obj = json::parse(response);

  if (response_obj.contains("content") && response_obj["content"].is_array() &&
      response_obj["content"].size() > 0) {
    std::string text = response_obj["content"][0]["text"];

    size_t json_start = text.find('{');
    if (json_start != std::string::npos) {
      size_t json_end = text.rfind('}');
      if (json_end != std::string::npos && json_end > json_start) {
        std::string json_str = text.substr(json_start, json_end - json_start + 1);

        json result = json::parse(json_str);
        if (result.contains("result")) {
          int64_t value = result["result"].get<int64_t>();
          if (value == 2) {
            llvm::errs() << "[KTDFOptimization] Agent returned correct value: 2\n";
            return value;
          } else {
            llvm::errs() << "[KTDFOptimization] WARNING: Agent returned " << value
                         << " instead of 2\n";
            return value;
          }
        }
      }
    }
  }

  llvm::report_fatal_error(llvm::StringRef("Response missing result field"));
  return 0;
}

}  // namespace scheduler
