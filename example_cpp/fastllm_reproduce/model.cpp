#include "basellm.h"
#include "fastllm.h"
#include "file_utils.hpp"
#include <memory>

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
}