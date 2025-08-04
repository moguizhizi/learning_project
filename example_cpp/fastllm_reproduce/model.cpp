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

    SafeTensors safetensors = LoadSafeTensors(modelPath);

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