// safetensors.cpp

#include "struct_space.hpp"
#include "enum_space.h"

SafeTensorItem::SafeTensorItem() {}

SafeTensorItem::~SafeTensorItem() {}

SafeTensorItem::SafeTensorItem(const std::string &tensorName, const std::string &fileName, uint64_t baseOffset, const json11::Json &config) {
    this->tensorName = tensorName;
    this->fileName = fileName;
    this->dtype = config["dtype"].string_value();

    for (auto &it : config["shape"].array_items()) {
        this->shape.push_back(it.ll_value());
        this->intShape.push_back(this->shape.back());
    }

    for (auto &it : config["data_offsets"].array_items()) {
        this->dataOffsets.push_back(baseOffset + it.ll_value());
    }

    for (auto &it : this->shape) {
        this->len = this->len * it;
    }

    this->bytesLen = this->dataOffsets[1] - this->dataOffsets[0];
}

SafeTensors::SafeTensors(const std::set<std::string> fileNames) {
    for (const std::string &fileName : fileNames) {
        FILE *file = fopen(fileName.c_str(), "rb");
        if (!file) {
            perror(("Line " + std::to_string(__LINE__) + ": fopen failed: " + fileName).c_str());
            exit(0);
        }

        uint64_t stlen;
        int ret = fread(&stlen, sizeof(uint64_t), 1, file);
        if (ret != 1) {
            perror(("Line " + std::to_string(__LINE__) + ": Failed read from: " + fileName).c_str());
            fclose(file);
            exit(0);
        }

        char *layers_info = new char[stlen + 5];
        layers_info[stlen] = 0;
        ret = fread(layers_info, 1, stlen, file);
        if (ret != stlen) {
            perror(("Line " + std::to_string(__LINE__) + ": Failed read from: " + fileName).c_str());
            fclose(file);
            exit(0);
        }

        std::string error;
        auto config = json11::Json::parse(layers_info, error);
        for (auto &it : config.object_items()) {
            if (it.first != "__metadata__") {
                std::cout << it.first << ":" << it.second.dump() << std::endl;
                this->itmeDict[it.first] = SafeTensorItem(it.first, fileName, stlen + 8, it.second);
            }
        }

        delete[] layers_info;
    }
}

std::vector<std::string> SafeTensors::GetSortedItemNames() {
    std::vector<std::pair<std::pair<std::string, uint64_t>, std::string>> v;
    for (auto &it : this->itmeDict) {

        std::string fileName = it.second.fileName;
        uint64_t baseOffset = it.second.dataOffsets[0];
        std::string tensorName = it.first;
        std::string dtype = it.second.dtype;
        std::vector<int> intShape = it.second.intShape;

        if (dtype != "BOOL" && intShape.size() > 0) {
            std::pair pair_1 = std::make_pair(fileName, baseOffset);
            std::pair pair_2 = std::make_pair(pair_1, tensorName);

            v.push_back(pair_2);
        }
    }

    std::sort(v.begin(), v.end());
    std::vector<std::string> ret;
    for (auto &it : v) {
        ret.push_back(it.second);
    }

    return ret;
}

void WeightMap::AddDict(const std::string &key, const std::string &value) { this->dicts[key] = value; }

WeightType WeightMap::GetWeightType(const std::string &key) {
    if (this->embeddingsNames.find(key) != this->embeddingsNames.end()) {
        return WeightType::EMBEDDING;
    }

    for (auto &linearName : this->linearNames) {
        int n = key.size();
        int m = linearName.size();
        std::vector<std::vector<bool>> f = std::vector<std::vector<bool>>(n + 1, std::vector<bool>(m + 1, 0));
        f[0][0] = 1;
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                if (f[i][j]) {
                    if ((i + 1 <= n) && (j + 1 <= m) && key[i] == linearName[j]) {
                        f[i + 1][j + 1] = 1;
                    }

                    if (j + 1 <= n && linearName[j] == '*') {
                        for (int l = i; l <= n; l++) {
                            f[l][j + 1] = 1;
                        }
                    }

                    if (i + 1 <= m && key[i] == '*') {
                        for (int l = j; l <= m; l++) {
                            f[i + 1][l] = 1;
                        }
                    }
                }
            }
        }
        if (f[m][n]) {
            return WeightType::LINEAR;
        }
    }

    return WeightType::AUTO;
}

WeightMergeRuleSingle::WeightMergeRuleSingle(const std::vector<std::string> &inputs, std::string output, std::string type)
    : inputs(inputs), output(output), type(type) {}

WeightMergeRule::WeightMergeRule(const std::vector<WeightMergeRuleSingle> &rules) {
    this->rules = rules;
    for (auto &rule : this->rules) {
        for (auto &input : rule.inputs) {
            this->allInputs.insert(input);
        }
    }
}

CudaMemoryBuffer::CudaMemoryBuffer() {}
CudaMemoryBuffer::CudaMemoryBuffer(void *data, size_t size, bool busy) : data(data), size(size), busy(busy) {}