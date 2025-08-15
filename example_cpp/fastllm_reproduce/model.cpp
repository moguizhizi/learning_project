#include "basellm.h"
#include "fastllm.h"
#include "file_utils.hpp"
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

std::unique_ptr<basellm> CreateLLMModelFromHF(const std::string &modelPath,
                                              DataType linearDataType,
                                              int groupCnt,
                                              bool skipTokenizer,
                                              const std::string &modelConfig,
                                              const std::string &loraPath,
                                              bool weightOnly,
                                              bool useMoeDataType,
                                              DataType moeDataType,
                                              int moeGroupCnt,
                                              const std::string &dtypeConfigString) {

    std::map<std::string, std::pair<std::string, std::string>> loraDicts;
    SafeTensors *loraTensors = nullptr;
    float loraScaling = 1.0f;
    LoadLoRA(loraPath, loraDicts, loraTensors, loraScaling);

    SafeTensors safeTensors = LoadSafeTensors(modelPath);

    bool isJsonModel = false;
    std::string configFile = modelPath + "config.json";
    std::string configError;
    auto config = json11::Json::parse(ReadAllFile(configFile), configError);

    std::string modelType = GetModelType(config, weightOnly, isJsonModel);
    basellm *model = CreateModelWithType(modelType);

    AddDictRecursion(model, "", config);

    bool isAwqModel = false;
    int awqGroupCnt = 128;
    CheckAWQModel(config, isAwqModel, awqGroupCnt);

    std::string generationConfigFile = modelPath + "generation_config.json";
    std::string generationConfigError;
    auto generation_config = json11::Json::parse(ReadAllFile(generationConfigFile), generationConfigError);
    SetEosTokenIds(model, config, generation_config);

    for (auto &it : generation_config.object_items()) {
        if (it.first.c_str() == "eos_token_id" && it.second.type() == json11::Json::ARRAY) {
            continue;
        }
        model->weight.AddDict(it.first.c_str(), it.second.is_string() ? it.second.string_value() : it.second.dump());
    }

    if (!skipTokenizer) {
        LoadLLMTokenizerFromHFToModel(modelPath, model);
    }

    model->InitParams();

    auto tensors = safeTensors.GetSortedItemNames();

    auto tensorMap = model->GetTensorMap(tensors, useMoeDataType, moeDataType);

    std::string dtype_config = "/home/project/learning_project/example_cpp/fastllm_reproduce/dtype_config.json";
    std::vector<std::pair<std::string, std::string>> dtypeRules = ParseDtypeRulesFromConfigFile(dtype_config);

    int cur = 0;
    long long totalBytes = 0;
    std::set<std::string> allWeightNames; // 所有创建了的weight name
    for (auto &tensorName : tensors) {
        auto &tensor = safeTensors.itmeDict[tensorName];
        DataType oriDataType = DataType::FLOAT32;
        for (auto &it : tensorMap[tensorName]) {
            std::string weightName = it.first;
            DataType dataType = it.second;
            allWeightNames.insert(weightName);

            int groupCnt = -1;
            dataType = ResolveAutoDataType(weightName, dtypeRules, dataType, groupCnt, linearDataType, oriDataType, tensor);

            if (it.second == DataType::DATA_AUTO_CONV) {
                std::vector<int> realshape = tensor.intShape;
                std::swap(realshape[0], realshape[1]);
                model->weight.AddEmptyWeight(weightName, realshape, dataType);
            } else if (isAwqModel && StringEndWith(tensorName, ".qweight")) {
                model->weight.AddEmptyWeight(weightName, {tensor.intShape[1] * 8, tensor.intShape[0]}, dataType);
            } else {
                model->weight.AddEmptyWeight(weightName, tensor.intShape, dataType);
            }
        }

        totalBytes += tensor.bytes;
        printf("Load %d \r", (++cur) * 100 / (int)safeTensors.itmeDict.size());
        fflush(stdout);
    }

    int thread_num = 16;
    int cur = 0;
    int start = 0;
    std::vector<std::pair<int, int>> parts;
    for (int i = 0; i < thread_num; i++) {
        long long now = 0;
        while (true) {
            if (now * thread_num >= totalBytes || start > tensors.size()) {
                break;
            } else {
                now += safeTensors.itmeDict[tensors[start]].bytes;
                start++;
            }
        }
        parts.push_back(std::make_pair(cur, start));
        cur = start;
    }
    parts.back().second = tensors.size();
    while (parts.size() < thread_num) {
        parts.push_back(std::make_pair(-1, -1));
    }

    std::set<std::string> allFinishName;
    std::vector<std::thread *> threads;
    std::mutex locker;
    int cnt = 0;
    int per = tensors.size() / thread_num;
    for (int i = 0; i < thread_num; i++) {
        threads.push_back(new std::thread(
            [&](int st, int end) {
                for (int j = st; j < end; j++) {
                    std::string &tensorName = tensors[j];
                    if (StringEndWith(tensorName, "_scale_inv") ||
                        (isAwqModel && (StringEndWith(tensorName, "qzeros") || StringEndWith(tensorName, "scales")))) {
                        continue;
                    }

                    locker.lock();
                    printf("Loading %d \r", (++cnt) * 100 / (int)tensorMap.size());
                    fflush(stdout);
                    locker.unlock();

                    std::string weightName = "";
                    DataType dataType = DataType::DATA_AUTO_NONE;
                    DataType oriDataType = DataType::FLOAT32;

                    for (auto &it : tensorMap[tensorName]) {
                        std::string scaleTensorName = "";
                        std::string qzeroTensorName = "";
                        weightName = it.first;
                        dataType = it.second;

                        SafeTensorItem tensor = safeTensors.itmeDict[tensorName];

                        int curGroupCnt = model->moeLinears.find(weightName) != model->moeLinears.end() ? moeGroupCnt : groupCnt;

                        dataType = ResolveAutoDataType(weightName, dtypeRules, dataType, groupCnt, linearDataType, oriDataType, tensor);

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
                            scaleTensorName = weightName + "_scale_inv";
                            if (safeTensors.itmeDict.find(scaleTensorName) == safeTensors.itmeDict.end()) {
                                scaleTensorName = "";
                            }
                        }

                        if (tensor.dtype == "I32" && isAwqModel && StringEndWith(tensorName, "qweight")) {
                            scaleTensorName = weightName + "scales";
                            qzeroTensorName = weightName + "qzeros";
                            oriDataType = DataType::FLOAT32;
                            AssertInFastLLM(safeTensors.itmeDict.find(scaleTensorName) != safeTensors.itmeDict.end() &&
                                                safeTensors.itmeDict.find(qzeroTensorName) != safeTensors.itmeDict.end(),
                                            "Tensor error: can't find AWQ scalse / qzeros.");
                            if (dataType == INT4_GROUP && groupCnt == awqGroupCnt) {
                                oriDataType = DataType::INT4_GROUP;
                            }
                        }

                        if (scaleTensorName == "") {
                            tensor.CreateBuffer(oriDataType);
                        } else if (!isAwqModel) {
                            auto &scaleTensor = safeTensors.itmeDict[scaleTensorName];
                            AssertInFastLLM(scaleTensor.dtype == "F32" || scaleTensor.dtype == "BF16",
                                            "Tensor scale error: scale's dtype should be F32 or BF16.");
                            scaleTensor.CreateBuffer(DataType::FLOAT32);
                            tensor.CreateBufferWithScale(dataType, scaleTensor);
                        } else {
                            auto &scaleTensor = safeTensors.itmeDict[scaleTensorName];
                            auto &qzeroTensor = safeTensors.itmeDict[qzeroTensorName];
                            scaleTensor.CreateBuffer(DataType::FLOAT32);
                            tensor.CreateBufferWithAWQ(oriDataType, scaleTensor, qzeroTensor);
                        }

                        ApplyLoRAWeight(weightName, loraDicts, loraTensors, tensor, oriDataType, loraScaling);

                        if (tensor.dtype == "fastllm") {
                            model->weight[weightName].CreateFromFastllmFormat(tensor.buffer, tensor.bytes);
                        } else {
                            if (it.second == DATA_AUTO_CONV) {
                                tensor.Transpose(oriDataType);
                            }
                            model->weight[weightName].CreateFromOriData(WeightType::AUTO,
                                                                        oriDataType,
                                                                        tensor.buffer,
                                                                        tensor.minsBuffer,
                                                                        tensor.scalesBuffer,
                                                                        curGroupCnt,
                                                                        tensor.blockK,
                                                                        tensor.blockM);
                        }

                        if (it.second == DATA_AUTO_LINEAR || it.second == DATA_AUTO_CONV)
                            model->weight[weightName].CalcWeightSum();

                        tensor.ClearBuffer();

                        locker.lock();
                        bool needMerge = false;
                        allFinishName.insert(weightName);
                        model->MergeWeightsFromRules(weightName, allWeightNames, allFinishName);
                        locker.unlock();

#if defined(USE_TFACC) || defined(USE_NUMA)
                        try {
                            std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
                            if (s != "" && s != "OFF") {
                                if (!needMerge && model->specialWeights.find(weightName) != model->specialWeights.end()) {
                                    locker.lock();
                                    model->weight.weight[weightName].weightSum.resize(1);
                                    RegisterFastllmData(&model->weight.weight[weightName], model->specialWeights[weightName]);
                                    locker.unlock();
                                }
                            }
                        } catch (...) {
                        }
#endif
                    }

                    locker.lock();
                    printf("Loading %d \r", (++cnt) * 100 / (int)tensorMap.size());
                    fflush(stdout);
                    locker.unlock();
                }
            },
            parts[i].first,
            parts[i].second));
    }
}