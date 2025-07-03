// file_utils.cpp
#include "file_utils.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

void ErrorInFastLLM(const std::string &error) {
    printf("FastLLM Error: %s\n", error.c_str());
    throw error;
}

std::string ReadAllFile(const std::string &fileName) {
    std::ifstream t(fileName.c_str(), std::ios::in);
    if (!t.good()) {
        ErrorInFastLLM("Read error: can't find \"" + fileName + "\".");
    }

    std::string ret((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    t.close();
    return ret;
}
