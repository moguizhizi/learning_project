#pragma once

#include "basellm.h"
#include <string>

basellm *CreateModelWithType(const std::string &model_type);
void AddDictRecursion(basellm *model, const std::string &prefix, const json11::Json &config);