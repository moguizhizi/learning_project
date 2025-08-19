// main.cpp
#include "struct_space.hpp"

int main() {
    std::set<std::string> fileNames = {"/home/temp/llm_model/Qwen/Qwen3-0.6B/model.safetensors"};
    SafeTensors safetensors(fileNames);
    safetensors.GetSortedItemNames();
}