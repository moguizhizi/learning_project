// safetensors.hpp

#pragma once

#include "enum_space.h"
#include "json11.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

struct SafeTensorItem {
    std::string tensorName;
    std::string fileName;
    std::string dtype;
    std::vector<uint64_t> shape;
    std::vector<int> intShape;
    std::vector<uint64_t> dataOffsets;

    uint64_t len = 1;
    uint64_t bytesLen = 1;

    SafeTensorItem();
    ~SafeTensorItem();
    SafeTensorItem(const std::string &tensorName, const std::string &fileName, uint64_t baseOffset, const json11::Json &config);
};

struct SafeTensors {
    std::map<std::string, SafeTensorItem> itmeDict;
    std::set<std::string> fileNames;

    SafeTensors(const std::set<std::string> fileNames);

    std::vector<std::string> GetSortedItemNames();
};

struct Tokenizer {};

struct WeightMap {
    std::set<std::string> embeddingsNames;
    std::set<std::string> linearNames;
    std::map<std::string, std::string> dicts;

    void AddDict(const std::string &key, const std::string &value);
    WeightType GetWeightType(const std::string &key);
};

struct WeightMergeRuleSingle {
    std::vector<std::string> inputs;
    std::string output;
    std::string type;

    WeightMergeRuleSingle(const std::vector<std::string> &inputs, std::string output, std::string type);
};

struct WeightMergeRule {
    std::vector<WeightMergeRuleSingle> rules;
    std::set<std::string> allInputs;

    WeightMergeRule(const std::vector<WeightMergeRuleSingle> &rules);
};

struct CudaMemoryBuffer {
    void *data;
    size_t size;
    bool busy;

    CudaMemoryBuffer();
    CudaMemoryBuffer(void *data, size_t size, bool busy);
};