#include <sstream>
#include <fstream>
#include "json11.hpp"
#include "utils.h" // 假设 utils.h 中定义了 ErrorInFastLLM 或其他工具函数
#include <iostream>

// 读取文件内容的函数
std::string ReadAllFile(const std::string &fileName) {
    std::ifstream t(fileName.c_str(), std::ios::in);
    // if (!t.good()) {
    //     ErrorInFastLLM("Read error: can't find \"" + fileName + "\".");
    // }

    std::string ret((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    t.close();
    return ret;
}

// 主函数
int main() {
    // 定义 JSON 解析错误字符串
    std::string loraConfigError;
    // 读取并解析 adapter_config.json 文件
    auto loraConfig = json11::Json::parse(ReadAllFile("/data1/temp/llm_lora/snshrivas10/sft-tiny-chatbot/adapter_config.json"), loraConfigError);

    // 检查 JSON 解析是否成功
    if (!loraConfigError.empty()) {
        std::cerr << "JSON parse error: " << loraConfigError << std::endl;
        return 1; // 返回非零表示错误
    }

    // 打印解析后的 JSON 内容
    std::cout << "Parsed JSON: " << loraConfig.dump() << std::endl;

    // 示例：访问 JSON 中的字段（假设 adapter_config.json 包含某些字段）
    if (loraConfig.is_object()) {
        std::cout << "JSON is an object with fields:" << std::endl;
        for (const auto &item : loraConfig.object_items()) {
            std::cout << "  " << item.first << ": " << item.second.dump() << std::endl;
        }
    } else {
        std::cout << "JSON is not an object." << std::endl;
    }

    float loraScaling;
    loraScaling = loraConfig["lora_alpha"].number_value() / loraConfig["r"].number_value();
    std::cout << loraScaling << std::endl;
    std::cout << loraConfig["lora_alpha"].number_value() << std::endl;
    std::cout << loraConfig["r"].number_value() << std::endl;


    return 0; // 成功返回 0
}