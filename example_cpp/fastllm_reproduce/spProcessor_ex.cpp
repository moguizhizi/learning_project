#include <iostream>
#include <sentencepiece_processor.h>
#include <string>
#include <vector>

int main() {
    // 创建并加载 tokenizer
    sentencepiece::SentencePieceProcessor sp;
    sp.Load("/home/llm_model/Shanghai_AI_Laboratory/internlm3-8b-instruct/tokenizer.model"); // 直接从文件加载，也可以用 LoadFromSerializedProto()

    // 要分词的文本
    std::string text = "Hello world, 你好世界！";

    // ===== 方法1：分成 token id =====
    std::vector<int> ids;
    sp.Encode(text, &ids);
    std::cout << "Token IDs: ";
    for (auto id : ids)
        std::cout << id << " ";
    std::cout << std::endl;

    // ===== 方法2：分成 subwords =====
    std::vector<std::string> pieces;
    sp.Encode(text, &pieces);
    std::cout << "Pieces: ";
    for (auto &p : pieces)
        std::cout << p << " | ";
    std::cout << std::endl;

    // ===== 方法3：反分词 =====
    std::string decoded;
    sp.Decode(ids, &decoded);
    std::cout << "Decoded: " << decoded << std::endl;

    return 0;
}
