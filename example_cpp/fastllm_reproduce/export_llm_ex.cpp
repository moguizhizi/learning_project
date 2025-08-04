#include "fastllm.h"
#include "file_utils.hpp"
#include "struct_space.hpp"
#include <cstring>
#include <thread>

int main() {

    // std::string loraPath = "/home/temp/llm_model/Qwen/Qwen3-8B/";
    std::string loraPath = "";

    std::map<std::string, std::pair<std::string, std::string>> loraDicts;
    SafeTensors *loraTensors = nullptr;
    float loraScaling;
    if (loraPath != "") {
        std::string path = loraPath;
        if (path.back() != '/' && path.back() != '\\') {
            path += "/";
        }
        loraTensors = new SafeTensors({path + "adapter_model.safetensors"});
        for (auto &it : loraTensors->GetSortedItemNames()) {
            if (it.size() >= 31 && it.substr(0, 17) == "base_model.model." &&
                (it.substr(it.size() - 14) == ".lora_A.weight" || it.substr(it.size() - 14) == ".lora_B.weight")) {
                std::string originalName = it.substr(17, it.size() - 31) + ".weight";
                if (it.substr(it.size() - 14) == ".lora_A.weight") {
                    loraDicts[originalName].first = it;
                } else {
                    loraDicts[originalName].second = it;
                }
            }
        }
        std::string loraConfigError;
        auto loraConfig = json11::Json::parse(ReadAllFile(path + "adapter_config.json"), loraConfigError);
        loraScaling = loraConfig["lora_alpha"].number_value() / loraConfig["r"].number_value();
    }

    std::string path = "/home/temp/llm_model/Qwen/Qwen3-0.6B/";
    std::string outputPath = "/home/temp/llm_model/fastllm/Qwen/Qwen3-8B/";
    std::string stIndexFile = path + "model.safetensors.index.json";

    std::set<std::string> stFiles;
    std::map<std::string, std::string> outputFileDict;
    if (!FileExists(stIndexFile)) {
        stFiles.insert(path + "model.safetensors");
        outputFileDict[path + "model.safetensors"] = outputPath + "model.safetensors";
    } else {
        std::string error;
        auto model_safetensors = json11::Json::parse(ReadAllFile(stIndexFile), error);
        auto stIndex = model_safetensors["weight_map"];

        for (auto &it : stIndex.object_items()) {
            stFiles.insert(path + it.second.string_value());
            outputFileDict[path + it.second.string_value()] = outputPath + it.second.string_value();
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
    if (useMoeDataType && model->moeLinears.size() > 0) {
        for (auto &it : tensorMap) {
            for (auto &weight : it.second) {
                if (model->moeLinears.find(weight.first) != model->moeLinears.end()) {
                    weight.second = moeDataType;
                }
            }
        }
    }

    std::string dtype_config = "/home/project/learning_project/example_cpp/fastllm_reproduce/dtype_config.json";
    std::vector<std::pair<std::string, std::string>> dtypeRules = ParseDtypeRulesFromConfigFile(dtype_config);

    DataType lineardtype = DataType::BFLOAT16;

    for (auto &file : safeTensors.fileNames) {
        std::map<std::string, Data> weights;
        std::vector<SafeTensorItem *> item;
        std::string outputFileName = outputFileDict[file];
        for (auto &it : safeTensors.itmeDict) {
            if (it.second.fileName == file) {
                item.push_back(&it.second);
            }
        }

        // 1.0 创建 weights
        json11::Json::object config;
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

                        int curGroupCnt = model->moeLinears.find(weightName) != model->moeLinears.end() ? moeGroupCnt : groupCnt;
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

                        if (tensor.dtype == "F8_E4M3" && (dataType == DataType::FP8_E4M3)) {
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

                        if (loraDicts.find(weightName) != loraDicts.end()) {
                            std::string loraA = loraDicts[weightName].first;
                            std::string loraB = loraDicts[weightName].second;

                            int inDim = loraTensors->itmeDict[loraA].intShape[1];
                            int outDim = loraTensors->itmeDict[loraB].intShape[0];
                            int lora = loraTensors->itmeDict[loraA].intShape[0];

                            loraTensors->itmeDict[loraA].CreateBuffer(DataType::FLOAT32);
                            loraTensors->itmeDict[loraB].CreateBuffer(DataType::FLOAT32);

                            float *weightA = (float *)(loraTensors->itmeDict[loraA].buffer);
                            float *weightB = (float *)(loraTensors->itmeDict[loraB].buffer);

                            std::vector<float> loraFactor;
                            loraFactor.resize(outDim * inDim, 0.0f);
                            for (int i = 0; i < outDim; i++) {
                                for (int j = 0; j < lora; j++) {
                                    for (int k = 0; k < inDim; k++) {
                                        loraFactor[i * inDim + k] += weightB[i * lora + j] * weightA[j * inDim + k];
                                    }
                                }
                            }

                            for (int i = 0; i < loraFactor.size(); i++) {
                                loraFactor[i] = loraFactor[i] * loraScaling;
                            }

                            loraTensors->itmeDict[loraA].ClearBuffer();
                            loraTensors->itmeDict[loraB].ClearBuffer();

                            if (oriDataType == DataType::BFLOAT16) {
                                uint16_t *fp16Weight = (uint16_t *)tensor.buffer;
                                for (int i = 0; i < loraFactor.size(); i++) {
                                    uint32_t now = fp16Weight[i] << 16;
                                    float newV = ((float *)&now)[0] + loraFactor[i];
                                    fp16Weight[i] = ((uint32_t *)&newV)[0] >> 16;
                                }
                            } else if (oriDataType == DataType::FLOAT16) {
                                uint16_t *fp16Weight = (uint16_t *)tensor.buffer;
                                for (int i = 0; i < loraFactor.size(); i++) {
                                    fp16Weight[i] = float_to_half(half_to_float(fp16Weight[i]) + loraFactor[i]);
                                }
                            } else if (oriDataType == DataType::FLOAT32) {
                                float *f32weight = (float *)tensor.buffer;
                                for (int i = 0; i < loraFactor.size(); i++) {
                                    f32weight[i] = f32weight[i] + loraFactor[i];
                                }
                            }
                        }

                        if (dataType == DataType::DATA_AUTO_CONV) {
                            tensor.Transpose(oriDataType);
                        }

                        weights[weightName].CreateFromOriData(WeightType::AUTO,
                                                              oriDataType,
                                                              tensor.buffer,
                                                              tensor.minsBuffer,
                                                              tensor.scalesBuffer,
                                                              curGroupCnt,
                                                              tensor.blockK,
                                                              tensor.blockM);
                        tensor.ClearBuffer();
                    }
                },
                st,
                end));
        }

        for (int i = 0; i < threads.size(); i++) {
            threads[i]->join();
            delete threads[i];
        }

        std::map<std::string, std::vector<long long>> offsets;
        long long currentOffset = 0;
        for (auto it : item) {
            std::string weightName = it->tensorName;
            DataType realType = weights[weightName].dataType;
            std::string dtype = "";
            if (realType == DataType::FLOAT16) {
                dtype = "F16";
            } else if (realType == DataType::FLOAT32) {
                dtype = "F32";
            } else if (realType == DataType::BFLOAT16) {
                dtype = "BF16";
            } else {
                dtype = "fastllm";
            }
            offsets[weightName] = {currentOffset, currentOffset + (long long)weights[weightName].GetFastllmFormateBytes()};
            currentOffset = currentOffset + (long long)weights[weightName].GetFastllmFormateBytes();
            config[weightName] = json11::Json::object{
                {"dtype", dtype},
                {"shape", json11::Json(weights[weightName].dims)},
                {"data_offsets", json11::Json(offsets[weightName])},
            };
        }

        std::string configString = json11::Json(config).dump();
        std::vector<uint8_t> bytes;
        bytes.resize(currentOffset);
        for (auto it : item) {
            std::string weightName = it->tensorName;
            if (StringEndWith(weightName, "_scale_inv")) {
                continue;
            }

            weights[weightName].ExportFastllmFormat(bytes.data() + offsets[weightName][0]);
        }

        FILE *outputFile = fopen(outputFileName.c_str(), "wb");
        uint64_t configLen = configString.size();
        fwrite(&configLen, sizeof(uint64_t), 1, outputFile);
        fwrite(configString.data(), 1, configString.size(), outputFile);
        fwrite(bytes.data(), 1, bytes.size(), outputFile);
        fclose(outputFile);
    }
    delete loraTensors;
    return 0;
}