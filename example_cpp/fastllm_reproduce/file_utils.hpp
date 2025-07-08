// file_utils.hpp
#pragma once
#include <string>

void ErrorInFastLLM(const std::string &error);
std::string ReadAllFile(const std::string &fileName);
void AssertInFastLLM(bool condition, const std::string &error);