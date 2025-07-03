// main.cpp
#include "struct_space.hpp"

int main() {
    std::vector<std::string> fileNames = {
        "/home/temp/llm_model/Qwen/Qwen2.5-VL-7B-Instruct/model-00004-of-00005.safetensors"
    };
    SafeTensors safetensors(fileNames);
}