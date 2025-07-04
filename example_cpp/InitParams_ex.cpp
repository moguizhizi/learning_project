#include "fastllm.h"
#include "file_utils.hpp"
#include "json11.hpp"
#include <string>

int main() {
    std::string config_path = "/home/temp/llm_model/Qwen/Qwen3-0___6B/config.json";
    std::string configError;
    std::string model_type;
    auto config = json11::Json::parse(ReadAllFile(config_path), configError);

    if (!config["model_type"].is_null()) {
        model_type = config["model_type"].string_value();
    } else {
        model_type = config["architectures"].array_items()[0].string_value();
    }

    basellm *model = CreateModelWithType(model_type);
    AddDictRecursion(model, "", config);

    
}