#include "basellm.h"
#include "fastllm.h"
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
}