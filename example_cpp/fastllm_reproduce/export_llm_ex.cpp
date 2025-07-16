#include "fastllm.h"
#include "file_utils.hpp"
#include "struct_space.hpp"
#include <cstring>

std::vector<std::pair<std::string, std::string>> ParseDtypeRulesFromConfigString(const std::string &dtypeConfigString) {
    std::vector<std::pair<std::string, std::string>> dtypeRules;

    if (!dtypeConfigString.empty()) {
        std::string error;
        auto dtypeConfig = json11::Json::parse(dtypeConfigString, error);

        if (!error.empty()) {
            std::cerr << "Parse dtype config failed.\n";
            std::cerr << "config = " << dtypeConfigString << "\n";
            std::cerr << "error = " << error << "\n";
        } else {
            for (const auto &item : dtypeConfig.array_items()) {
                std::string key = item["key"].string_value();
                std::string dtype = item["dtype"].string_value();
                dtypeRules.emplace_back(key, dtype);
            }
        }
    }

    if (!dtypeRules.empty()) {
        std::cout << "Dtype rules:\n";
        for (const auto &rule : dtypeRules) {
            std::cout << rule.first << ": " << rule.second << "\n";
        }
    }

    return dtypeRules;
}

int main() {

    std::string path = "/home/temp/llm_model/Qwen/Qwen3-8B/";
    std::string stIndexFile = path + "model.safetensors.index.json";

    std::set<std::string> stFiles;
    if (!FileExists(stIndexFile)) {
        stFiles.insert(path + "model.safetensors");
    } else {
        std::string error;
        auto model_safetensors = json11::Json::parse(ReadAllFile(stIndexFile), error);
        auto stIndex = model_safetensors["weight_map"];

        for (auto &it : stIndex.object_items()) {
            stFiles.insert(path + it.second.string_value());
        }
    }

    SafeTensors safetensors(stFiles);

    // 2. 创建网络基本信息
    std::string configFile = path + "config.json";
    std::string configError;
    std::string modelType;
    auto config = json11::Json::parse(ReadAllFile(configFile), configError);

    if (!config["model_type"].is_null()) {
        modelType = config["model_type"].string_value();
    } else {
        modelType = config["architectures"].array_items()[0].string_value();
    }

    basellm *model = CreateModelWithType(modelType);
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

    std::string dtypeConfigString = "";
    std::vector<std::pair<std::string, std::string>> dtypeRules = ParseDtypeRulesFromConfigString(dtypeConfigString);

    DataType lineardtype = DataType::BFLOAT16;

    for (auto &file : safetensors.fileNames) {
        std::map<std::string, Data> weights;
        std::vector<SafeTensorItem *> item;
        for (auto &it : safetensors.itmeDict) {
            if (it.second.fileName == file) {
                item.push_back(&it.second);
            }
        }

        // 1.0 创建 weights
        for (auto &it : item) {
            auto &tensor = *it;
            std::string weightName = tensor.tensorName;
            DataType oriDataType = DataType::FLOAT32;
            DataType dataType = tensorMap[weightName][0].second;

            if ((dataType == DataType::DATA_AUTO_LINEAR || dataType == DataType::DATA_AUTO_CONV) && dtypeRules.size() > 0) {
                int groupCnt = -1;
                ParseDataType(weightName, dtypeRules, dataType, groupCnt);

                if (tensor.dtype != "FP8_E4M3" && dataType == DataType::FP8_E4M3) {
                    dataType = DataType::FLOAT16;
                }
            }

            if (dataType >= DataType::DATA_AUTO_NONE) {
                dataType = (dataType == DataType::DATA_AUTO_LINEAR || dataType == DataType::DATA_AUTO_CONV) ? lineardtype : oriDataType;

                if (tensor.dtype != "FP8_E4M3" && dataType == DataType::FP8_E4M3) {
                    dataType = DataType::FLOAT16;
                }
            }

            if (dataType == DataType::DATA_AUTO_CONV) {
                std::vector<int> realshape = it->intShape;
                std::swap(realshape[0], realshape[1]);
                weights[weightName] = Data(dataType, realshape);
            } else {
                weights[weightName] = Data(dataType, it->intShape);
            }
        }
    }
}