
#include "fastllm.h"
#include "file_utils.hpp"
#include "struct_space.hpp"
#include <set>
#include <string>
#include <vector>

int main() {
    std::set<std::string> fileNames = {"/home/temp/llm_model/nm-testing/Qwen2___5-VL-72B-Instruct-quantized___w8a8/model-00014-of-00016.safetensors"};
    SafeTensors safetensors(fileNames);

    std::string config_path = "/home/temp/llm_model/nm-testing/Qwen2___5-VL-72B-Instruct-quantized___w8a8/config.json";
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

    model->InitParams();

    std::vector<std::string> tensors = safetensors.GetSortedItemNames();

    std::map<std::string, std::vector<std::pair<std::string, DataType>>> tensorMap = model->GetTensorMap(tensors);

    bool useMoeDataType = true;
    DataType moeDataType = DataType::FLOAT16;

    if (useMoeDataType && model->moelinears.size() > 0) {
        for (auto &it : tensorMap) {
            for (auto &weight : it.second) {
                if (model->moelinears.find(weight.first) != model->moelinears.end()) {
                    weight.second = moeDataType;
                }
            }
        }
    }
}