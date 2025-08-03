#include "fastllm.h"
#include "file_utils.hpp"

int main() {
    std::vector<std::string> lines, line;
    SplitString(ReadAllFile("/home/temp/llm_model/ZhipuAI/LongCite-glm4-9b/tokenizer.model"), {'\r', '\n'}, lines);
    for (int i = 0; i < lines.size(); i++) {
        SplitString(lines[i], {' '}, line);
        std::cout << Base64Decode(line[0]) << std::endl;
        std::cout << line[1].c_str() << std::endl;
        // model->weight.AddTokenizerWord(Base64Decode(line[0]), atoi(line[1].c_str()), 1.0f);
    }
    return 0;
}