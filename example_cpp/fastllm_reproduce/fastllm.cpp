#include "fastllm.h"
#include "file_utils.hpp"
#include "qwen3.h"
#include <algorithm>
#include <cstring>
#include <regex>

basellm *CreateModelWithType(const std::string &model_type) {
    basellm *model = nullptr;
    if (model_type == "qwen3") {
        model = new Qwen3Model();
    }

    return model;
}

void AddDictRecursion(basellm *model, const std::string &prefix, const json11::Json &config) {
    for (auto &it : config.object_items()) {
        if (it.second.is_object()) {
            AddDictRecursion(model, prefix + it.first + ".", it.second);
        } else {
            model->weight.AddDict(prefix + it.first, it.second.is_string() ? it.second.string_value() : it.second.dump());
        }
    }
}

bool StringEndWith(const std::string &s, const std::string &end) { return s.size() >= end.size() && s.substr(s.size() - end.size()) == end; }
bool StringStartWith(const std::string &s, const std::string &end) { return s.size() >= end.size() && s.substr(0, end.size()) == end; }
void ParseDataType(std::string weightName, const std::vector<std::pair<std::string, std::string>> &dtypeRules, DataType &dataType, int &groupCnt) {
    std::string matchedType = "";
    for (int i = 0; i < dtypeRules.size(); i++) {
        std::regex pattern(dtypeRules[i].first);
        if (std::regex_search(weightName, pattern)) {
            matchedType = dtypeRules[i].second;
        }
    }
    transform(matchedType.begin(), matchedType.end(), matchedType.begin(), ::tolower);
    if (matchedType != "") {
        for (auto &it : dataTypeNames) {
            for (auto &dataTypeName : it.second) {
                if (DefaultGroupCnts.find(it.first) != DefaultGroupCnts.end()) {
                    if (StringStartWith(matchedType, dataTypeName)) {
                        dataType = it.first;
                        if (matchedType != dataTypeName) {
                            groupCnt = std::atoi(matchedType.substr(dataTypeName.size()).c_str());
                        } else {
                            groupCnt = DefaultGroupCnts[it.first];
                        }
                    }
                } else {
                    if (matchedType == dataTypeName) {
                        dataType = it.first;
                    }
                }
            }
        }
    }
}

uint32_t as_uint(const float x) { return *(uint32_t *)&x; }
float as_float(const uint32_t x) { return *(float *)&x; }

float half_to_float(const uint16_t x) {         // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0,
                                                // +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint32_t e = (x & 0x7C00) >> 10;      // exponent
    const uint32_t m = (x & 0x03FF) << 13;      // mantissa
    const uint32_t v = as_uint((float)m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
    return as_float((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
                    ((e == 0) & (m != 0)) * ((v - 37) << 23 | ((m << (150 - v)) & 0x007FE000))); // sign : normalized : denormalized
}

uint16_t float_to_half(const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5,
                                        // +-5.9604645E-8, 3.311 digits
    const uint32_t b = as_uint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
    const uint32_t e = (b & 0x7F800000) >> 23;  // exponent
    const uint32_t m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    return (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
           ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) | (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
}

void ConvertDataType(uint8_t *src, DataType srcDtype, uint8_t *dst, DataType dstDtype, uint64_t len) {
    if (srcDtype == dstDtype) {
        int unitSize = 4;
        if (dstDtype == DataType::FLOAT32) {
            unitSize = 4;
        } else if (dstDtype == DataType::BFLOAT16 || dstDtype == DataType::FLOAT16) {
            unitSize = 2;
        } else {
            ErrorInFastLLM("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
        }
        std::memcpy(dst, src, unitSize * len);
    } else if (srcDtype == DataType::FP8_E4M3 && dstDtype == DataType::FLOAT16) {
        ErrorInFastLLM("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
    } else if (srcDtype == DataType::BFLOAT16 && dstDtype == DataType::FLOAT32) {
        uint16_t *u16src = (uint16_t *)src;
        uint16_t *u16dst = (uint16_t *)dst;

        for (int i = 0; i < len; i++) {
            u16dst[2 * i] = 0;
            u16dst[2 * i + 1] = u16src[i];
        }
    } else if (srcDtype == DataType::FLOAT16 && dstDtype == DataType::FLOAT32) {
        uint16_t *u16src = (uint16_t *)src;
        float *fdst = (float *)dst;

        for (int i = 0; i < len; i++) {
            fdst[i] = half_to_float(u16src[i]);
        }
    } else {
        ErrorInFastLLM("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
    }
}

void Transpose4x4(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
    if (n < 4 || m < 4) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                pDst[j * dstStride + i] = pSrc[i * srcStride + j];
            }
        }

        return;
    }
}

void TransposeF32(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
    int per = 4;
    for (int i = 0; i < n; i += per) {
        for (int j = 0; j < m; j += per) {
            Transpose4x4(pDst + j * dstStride + i, pSrc + i * srcStride + j, dstStride, srcStride, std::min(per, n - i), std::min(per, m - j));
        }
    }
}

std::string GetModelType(const json11::Json &config, bool weightOnly, bool isJsonModel) {
    std::string modelType;

    if (weightOnly) {
        modelType = "qwen";
    } else if (isJsonModel) {
        modelType = "fastllmJson";
    } else {
        if (!config["model_type"].is_null()) {
            modelType = config["model_type"].string_value();
        } else if (!config["architectures"].is_null() && !config["architectures"].array_items().empty()) {
            modelType = config["architectures"].array_items()[0].string_value();
        }

        // 特例处理 InternLM2
        if (!config["architectures"].is_null()) {
            std::string arch = config["architectures"].array_items()[0].string_value();
            if (arch == "InternLM2ForCausalLM") {
                modelType = "internlm2";
            }
        }
    }

    return modelType;
}

void CheckAWQModel(const json11::Json &config, bool &isAwqModel, int &awqGroupCnt) {
    isAwqModel = false;
    awqGroupCnt = 128;

    if (!config["quantization_config"].is_null() && config["quantization_config"]["quant_method"] == "awq") {
        auto qconfig = config["quantization_config"];

        AssertInFastLLM(qconfig["quant_method"] == "awq" && qconfig["bits"] == 4 && qconfig["version"] == "gemm" &&
                            qconfig["zero_point"].bool_value(),
                        "Config error: only 4bits AWQ with zero point and gemm version is supported.");

        isAwqModel = true;
        awqGroupCnt = qconfig["group_size"].int_value();
    }
}

void SetEosTokenIds(basellm *model, const json11::Json &config, const json11::Json &generation_config) {
    if (config["eos_token_id"].is_array()) {
        for (const auto &it : config["eos_token_id"].array_items()) {
            model->eos_token_ids.insert(it.int_value());
        }
    } else if (!config["eos_token_id"].is_null()) {
        model->eos_token_id = config["eos_token_id"].int_value();
    }

    if (generation_config["eos_token_id"].is_array()) {
        for (const auto &it : generation_config["eos_token_id"].array_items()) {
            model->eos_token_ids.insert(it.int_value());
        }
    }
}

void SplitString(const std::string &str, const std::set<char> &chars, std::vector<std::string> &ret) {
    ret.clear();
    std::string now = "";
    for (int i = 0; i < str.size(); i++) {
        if (chars.find(str[i]) == chars.end()) {
            now += str[i];
        } else {
            if (now != "") {
                ret.push_back(now);
                now = "";
            }
        }
    }
    if (now != "") {
        ret.push_back(now);
    }
}

std::string Base64Decode(const std::string &encoded) {
    static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                            "abcdefghijklmnopqrstuvwxyz"
                                            "0123456789+/";
    int in_len = encoded.size();
    int i = 0, j = 0, in_ = 0;
    char char_array_4[4], char_array_3[3];
    std::string ret = "";

    while (in_len-- && (encoded[in_] != '=')) {
        char_array_4[i++] = encoded[in_];
        in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++)
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
            for (i = 0; (i < 3); i++)
                ret.push_back(char_array_3[i]);
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++)
            char_array_4[j] = 0;

        for (j = 0; j < 4; j++)
            char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++)
            ret.push_back(char_array_3[j]);
    }

    return ret;
}

void LoadLLMTokenizerFromHFToModel(const std::string &path, basellm *model) {
    std::string error;
    std::string tokenizerConfigFile = path + "tokenizer_config.json";
    auto tokenizerConfig = json11::Json::parse(ReadAllFile(tokenizerConfigFile), error);
    model->weight.tokenizer.SetTokenizerConfig(tokenizerConfig);
    model->weight.tokenizer.SetChatTemplate();
    std::string chatTemplate = model->weight.tokenizer.chatTemplate;
    if (!chatTemplate.empty() && model->weight.dicts.find("chat_template") == model->weight.dicts.end()) {
        model->weight.AddDict("chat_template", chatTemplate);
    }

    std::string tokenizerClass = tokenizerConfig["tokenizer_class"].string_value();
    if (tokenizerClass == "PreTrainedTokenizerFast" || tokenizerClass == "LlamaTokenizerFast" || tokenizerClass == "Qwen2Tokenizer" ||
        tokenizerClass == "BloomTokenizer" || tokenizerClass == "LlamaTokenizer" || tokenizerClass == "CodeLlamaTokenizer" ||
        tokenizerClass == "MiniCPMTokenizer") {
        std::string tokenizerFile = path + "tokenizer.json";
        if (!FileExists(tokenizerFile)) {
            ErrorInFastLLM("Model with a supported tokenizer_class: " + tokenizerClass + "，but has no \"tokenizer.json\"!");
        }

        auto tokenizer = json11::Json::parse(ReadAllFile(tokenizerFile), error);
        for (auto &it : tokenizer["model"]["vocab"].object_items()) {
            model->weight.AddTokenizerWord(it.first, it.second.int_value(), 1.0f);
        }

        std::map<std::string, int> specialTokenMap;
        for (auto &it : tokenizer["model"]["added_tokens"].array_items()) {
            specialTokenMap[it["content"].string_value()] = it["id"].int_value();
        }

        if (!specialTokenMap.empty())
            model->weight.AddDict("tokenizer_has_special_tokens", "1");

        model->weight.tokenizer.SetSpecialTokens(specialTokenMap);

        if (!tokenizer["decoder"].is_null() && !tokenizer["decoder"]["type"].is_null() && tokenizer["decoder"]["type"] == "ByteLevel") {
            model->weight.tokenizer.byteAsChar = true;
            model->weight.AddDict("tokenizer_byte_as_char", "True");
        }
    } else if (tokenizerClass == "ChatGLM4Tokenizer") {
        std::vector<std::string> lines, line;
        SplitString(ReadAllFile(path + "tokenizer.model"), {'\n'}, lines);
        for (int i = 0; i < lines.size(); i++) {
            SplitString(lines[i], {' '}, line);
            model->weight.AddTokenizerWord(Base64Decode(line[0]), atoi(line[1].c_str()), 1.0f);
        }

        std::map<std::string, int> specialTokenMap;
        for (auto &it : tokenizerConfig["added_tokens_decoder"].object_items()) {
            specialTokenMap[it.second["content"].string_value()] = atoi(it.first.c_str());
        }

        if (!specialTokenMap.empty())
            model->weight.AddDict("tokenizer_has_special_tokens", "1");

        model->weight.tokenizer.SetSpecialTokens(specialTokenMap);
        model->weight.AddDict("tokenizer_class", tokenizerClass);
        // ChatGLM采用拼接token的方法，需要强行指定分割词的TokenID
        model->pre_prompt = "[gMASK]<sop>";
        model->user_role = ("<FLM_FIX_TOKEN_" + std::to_string(model->weight.tokenizer.GetTokenId("<|user|>")) + ">\n");
        model->bot_role = ("<FLM_FIX_TOKEN_" + std::to_string(model->weight.tokenizer.GetTokenId("<|assistant|>")) + ">\n");
        model->history_sep = "";
        model->weight.tokenizer.type = Tokenizer::TokenizerType::QWEN;
        model->weight.tokenizer.chatTemplate = "";

    } else if (tokenizerClass == "QWenTokenizer") {
        // Qwen用的分词
        std::vector<std::string> lines, line;
        SplitString(ReadAllFile(path + "qwen.tiktoken"), {'\n'}, lines);
        for (int i = 0; i < lines.size(); i++) {
            SplitString(lines[i], {' '}, line);
            model->weight.AddTokenizerWord(Base64Decode(line[0]), atoi(line[1].c_str()), 1.0f);
        }
        model->weight.tokenizer.type = Tokenizer::TokenizerType::QWEN;
        model->weight.tokenizer.chatTemplate = "";
        model->weight.dicts["im_start_id"] = std::to_string(lines.size() + 1);
        model->weight.dicts["im_end_id"] = std::to_string(lines.size() + 2);
    } else {
        ErrorInFastLLM("Unsupport tokenizer_class: " + tokenizerClass);
    }
}

void LoadLoRA(const std::string &loraPath,
              std::map<std::string, std::pair<std::string, std::string>> &loraDicts,
              SafeTensors *&loraTensors,
              float &loraScaling) {
    loraDicts.clear();
    loraTensors = nullptr;
    loraScaling = 1.0f;

    if (!loraPath.empty()) {
        std::string path = loraPath;
        if (path.back() != '/' && path.back() != '\\') {
            path += "/";
        }

        loraTensors = new SafeTensors({path + "adapter_model.safetensors"});

        for (const auto &it : loraTensors->GetSortedItemNames()) {
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

        if (loraConfigError.empty()) {
            float loraAlpha = loraConfig["lora_alpha"].number_value();
            float r = loraConfig["r"].number_value();
            if (r != 0) {
                loraScaling = loraAlpha / r;
            }
        }
    }
}

SafeTensors LoadSafeTensors(const std::string &path) {
    std::set<std::string> stFiles;
    std::string stIndexFile = path + "model.safetensors.index.json";
    std::string error;

    if (!FileExists(stIndexFile)) {
        stFiles.insert(path + "model.safetensors");
    } else {
        auto stIndex = json11::Json::parse(ReadAllFile(stIndexFile), error)["weight_map"];
        for (const auto &it : stIndex.object_items()) {
            stFiles.insert(path + it.second.string_value());
        }
    }

    return SafeTensors(stFiles);
}

std::vector<std::pair<std::string, std::string>> ParseDtypeRulesFromConfigFile(const std::string &configPath) {
    std::vector<std::pair<std::string, std::string>> dtypeRules;

    std::string dtypeConfigString = ReadAllFile(configPath);
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

DataType ResolveAutoDataType(const std::string &weightName,
                             const std::vector<std::pair<std::string, std::string>> &dtypeRules,
                             DataType dataType,
                             int curGroupCnt,
                             DataType linearDataType,
                             DataType oriDataType,
                             const SafeTensorItem &tensor) {
    // 判断是否是 AUTO_LINEAR 或 AUTO_CONV 类型，且有规则需要解析
    if ((dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) && !dtypeRules.empty()) {
        // 尝试根据 dtypeRules 推断 dataType
        ParseDataType(weightName, dtypeRules, dataType, curGroupCnt);

        // 如果推断结果是 FP8_E4M3 但 tensor 实际不是 FP8_E4M3，则降级为 FLOAT16
        if (tensor.dtype != "FP8_E4M3" && dataType == DataType::FP8_E4M3) {
            dataType = DataType::FLOAT16;
        }
    }

    // 如果是自动类型（DATA_AUTO_*），替换为具体的 linearDataType 或 oriDataType
    if (dataType >= DATA_AUTO_NONE) {
        dataType = (dataType == DATA_AUTO_LINEAR || dataType == DATA_AUTO_CONV) ? linearDataType : oriDataType;

        // 替换后仍需检查是否为不兼容的 FP8_E4M3
        if (tensor.dtype != "FP8_E4M3" && dataType == DataType::FP8_E4M3) {
            dataType = DataType::FLOAT16;
        }
    }

    return dataType;
}

void ApplyLoRAWeight(const std::string &weightName,
                     const std::map<std::string, std::pair<std::string, std::string>> &loraDicts,
                     SafeTensors *loraTensors,
                     SafeTensorItem &tensor,
                     DataType oriDataType,
                     float loraScaling) {

    auto it = loraDicts.find(weightName);
    if (it == loraDicts.end()) {
        return; // 没有对应的 LoRA 权重，直接退出
    }

    // 获取 LoRA A/B 权重的 key
    const std::string &loraA = it->second.first;
    const std::string &loraB = it->second.second;

    // 获取维度信息
    int inDim = loraTensors->itmeDict[loraA].intShape[1];
    int outDim = loraTensors->itmeDict[loraB].intShape[0];
    int lora = loraTensors->itmeDict[loraA].intShape[0];

    // 创建缓冲区
    loraTensors->itmeDict[loraA].CreateBuffer(DataType::FLOAT32);
    loraTensors->itmeDict[loraB].CreateBuffer(DataType::FLOAT32);

    float *weightA = (float *)(loraTensors->itmeDict[loraA].buffer);
    float *weightB = (float *)(loraTensors->itmeDict[loraB].buffer);

    // 计算 LoRA 矩阵乘法结果
    std::vector<float> loraFactor(outDim * inDim, 0.0f);
    for (int i = 0; i < outDim; i++) {
        for (int j = 0; j < lora; j++) {
            for (int k = 0; k < inDim; k++) {
                loraFactor[i * inDim + k] += weightB[i * lora + j] * weightA[j * inDim + k];
            }
        }
    }

    // 缩放 LoRA 权重
    for (float &v : loraFactor) {
        v *= loraScaling;
    }

    // 释放临时缓冲区
    loraTensors->itmeDict[loraA].ClearBuffer();
    loraTensors->itmeDict[loraB].ClearBuffer();

    // 按原始数据类型合并到 tensor
    if (oriDataType == DataType::BFLOAT16) {
        uint16_t *fp16Weight = (uint16_t *)tensor.buffer;
        for (size_t i = 0; i < loraFactor.size(); i++) {
            uint32_t now = fp16Weight[i] << 16;
            float newV = ((float *)&now)[0] + loraFactor[i];
            fp16Weight[i] = ((uint32_t *)&newV)[0] >> 16;
        }
    } else if (oriDataType == DataType::FLOAT16) {
        uint16_t *fp16Weight = (uint16_t *)tensor.buffer;
        for (size_t i = 0; i < loraFactor.size(); i++) {
            fp16Weight[i] = float_to_half(half_to_float(fp16Weight[i]) + loraFactor[i]);
        }
    } else if (oriDataType == DataType::FLOAT32) {
        float *f32weight = (float *)tensor.buffer;
        for (size_t i = 0; i < loraFactor.size(); i++) {
            f32weight[i] += loraFactor[i];
        }
    }
}

void barrier() {
#ifdef __aarch64__
    asm volatile("dmb ish");
#elif defined(_WIN32) || defined(_WIN64)
    MemoryBarrier();
#else
    __asm__ __volatile__("" : : : "memory");
#endif
}
