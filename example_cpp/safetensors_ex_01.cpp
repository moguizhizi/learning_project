// main.cpp
#include "struct_space.hpp"

int main() {
    std::set<std::string> fileNames = {
        "/home/temp/llm_model/Qwen/Qwen2.5-VL-7B-Instruct/model-00001-of-00005.safetensors"
    };
    SafeTensors safetensors(fileNames);
    safetensors.GetSortedItemNames();
}