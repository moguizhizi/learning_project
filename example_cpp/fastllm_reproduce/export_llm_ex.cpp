#include "fastllm.h"
#include "file_utils.hpp"
#include "struct_space.hpp"
#include <cstring>
#include <thread>

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

    SafeTensors safeTensors(stFiles);

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

    std::vector<std::string> tensors = safeTensors.GetSortedItemNames();
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

    for (auto &file : safeTensors.fileNames) {
        std::map<std::string, Data> weights;
        std::vector<SafeTensorItem *> item;
        for (auto &it : safeTensors.itmeDict) {
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

        std::vector<std::thread *> threads;
        int thread_num = 16;
        int per = item.size() / thread_num;
        int moeGroupCnt = 16;
        int groupCnt = 128;

        for (int i = 0; i < thread_num; i++) {
            int st = per * i;
            int end = (i == thread_num - 1) ? item.size() : per * (i + 1);
            threads.push_back(new std::thread(
                [&](int st, int end) {
                    for (int j = st; j < end; j++) {
                        auto &tensor = *item[j];
                        if (StringEndWith(tensor.tensorName, "_scale_inv")) {
                            continue;
                        }

                        std::string weightName = tensor.tensorName;
                        std::string scaleTensorName = "";
                        DataType oriDataType = DataType::FLOAT32;
                        DataType dataType = tensorMap[weightName][0].second;

                        int curGroupCnt = model->moelinears.find(weightName) != model->moelinears.end() ? moeGroupCnt : groupCnt;
                        if ((dataType == DataType::DATA_AUTO_LINEAR || dataType == DataType::DATA_AUTO_CONV) && dtypeRules.size() > 0) {
                            int groupCnt = -1;
                            ParseDataType(weightName, dtypeRules, dataType, groupCnt);
                        }

                        if (dataType >= DataType::DATA_AUTO_NONE) {
                            dataType = (dataType == DataType::DATA_AUTO_LINEAR || dataType == DataType::DATA_AUTO_CONV) ? lineardtype : oriDataType;
                        }

                        if (tensor.dtype == "BF16" && (dataType == DataType::FLOAT16 || dataType == DataType::INT8 ||
                                                       dataType == DataType::INT4_GROUP || dataType == DataType::INT4_NOZERO)) {
                            oriDataType = DataType::BFLOAT16;
                        }

                        if (tensor.dtype == "F16" && dataType == DataType::FLOAT16) {
                            oriDataType = DataType::FLOAT16;
                        }

                        if (tensor.dtype == "F8_E4M3" &&
                            (dataType == DataType::FLOAT32 || dataType == DataType::FLOAT16 || dataType == DataType::INT8 ||
                             dataType == DataType::INT4_GROUP || dataType == DataType::INT4_NOZERO)) {
                            oriDataType = DataType::FLOAT32;
                            scaleTensorName = weightName + "_scale_inv";
                            if (safeTensors.itmeDict.find(scaleTensorName) == safeTensors.itmeDict.end()) {
                                scaleTensorName = "";
                            }
                        }

                        if (tensor.dtype == "F8_E4M3" && (dataType == FP8_E4M3)) {
                            oriDataType = DataType::FP8_E4M3;
                            scaleTensorName = tensor.tensorName + "_scale_inv";
                            if (safeTensors.itmeDict.find(scaleTensorName) == safeTensors.itmeDict.end()) {
                                scaleTensorName = "";
                            }
                        }

                        if (scaleTensorName == "") {
                            tensor.CreateBuffer(oriDataType);
                        } else {
                            auto &scaleTensor = safeTensors.itmeDict[scaleTensorName];
                            AssertInFastLLM(scaleTensor.dtype == "F32" || scaleTensor.dtype == "BF16",
                                            "Tensor scale error: scale's dtype should be F32 or BF16.");
                            scaleTensor.CreateBuffer(DataType::FLOAT32);
                            tensor.CreateBufferWithScale(dataType, scaleTensor);
                        }
                    }
                },
                st,
                end));
        }
    }
}