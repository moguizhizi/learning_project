// safetensors.cpp

#include "struct_space.hpp"
#include "enum_space.h"
#include "fastllm.h"
#include "file_utils.hpp"
#include <cstring>

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

    this->bytes = this->dataOffsets[1] - this->dataOffsets[0];
}

void SafeTensorItem::ClearBuffer() {
    delete[] this->buffer;
    this->buffer = nullptr;
    delete[] this->minsBuffer;
    this->minsBuffer = nullptr;
    delete[] this->scalesBuffer;
    this->scalesBuffer = nullptr;
}

void SafeTensorItem::CreateBuffer(DataType dstType) {
    FILE *fi = fopen(this->fileName.c_str(), "rb");
#if defined(_WIN32) || defined(_WIN64)
    _fseeki64(fi, this->dataOffsets[0], 0);
#else
    fseek(fi, this->dataOffsets[0], 0);
#endif

    this->ClearBuffer();
    int ret;
    DataType srcType;
    if (this->dtype == "fastllm") {
        this->buffer = new uint8_t[this->bytes];
        ret = fread(this->buffer, 1, this->bytes, fi);
        fclose(fi);
        return;
    } else if (this->dtype == "F32") {
        srcType = DataType::FLOAT32;
        if (dstType != DataType::FLOAT32) {
            ErrorInFastLLM("SafeTensorItem.CreateBuffer: unsupport src dtype " + this->dtype + "\n");
        }
    } else if (this->dtype == "F16") {
        srcType = DataType::FLOAT16;
    } else if (this->dtype == "BF16") {
        srcType = DataType::BFLOAT16;
    } else if (this->dtype == "F8E4M3") {
        srcType = DataType::FP8_E4M3;
    } else if (this->dtype == "I64") {
        printf("skip I64 tensor %s\n", this->tensorName.c_str());
        return;
    } else {
        ErrorInFastLLM("SafeTensorItem.CreateBuffer: unsupport src dtype " + this->dtype + "\n");
    }

    int unitSize = 4;
    if (dstType == DataType::FLOAT32) {
        unitSize = 4;
    } else if (dstType == DataType::BFLOAT16 || dstType == DataType::FLOAT16) {
        unitSize = 2;
    } else {
        ErrorInFastLLM("SafeTensorItem.CreateBuffer: unsupport dst dtype " + std::to_string(dstType) + "\n");
    }

    this->buffer = new uint8_t[unitSize * (size_t)this->len];
    if (srcType == dstType) {
        ret = fread(this->buffer, 1, this->bytes, fi);
    } else {
        uint8_t *ori = new uint8_t[this->bytes];
        ret = fread(ori, 1, this->bytes, fi);
        ConvertDataType(ori, srcType, this->buffer, dstType, this->len);
        delete[] ori;
    }

    fclose(fi);
}

FP8E4M3ToFP32Manager fp8e4m3tofp32;
void SafeTensorItem::CreateBufferWithScale(DataType dstType, SafeTensorItem &scale) {
    AssertInFastLLM(this->shape.size() == 2 && scale.shape.size() == 2, "CreateBufferWithScale error: shape.size() should be 2.");
    DataType srcType;
    if (this->dtype == "F8_E4M3") {
        srcType = DataType::FP8_E4M3;
    } else {
        ErrorInFastLLM("CreateBufferWithScale error: dtype should be FP8_E4M3");
    }

    int n = this->intShape[0];
    int m = this->intShape[1];
    int ns = scale.intShape[0];
    int ms = scale.intShape[1];

    int blockN = n / ns;
    int blockM = m / ms;

    while (blockN & -blockN != blockN) {
        blockN++;
    }

    while (blockM & -blockM != blockM) {
        blockM++;
    }

    this->ClearBuffer();

    FILE *fi = fopen(this->fileName.c_str(), "rb");
#if defined(_WIN32) || defined(_WIN64)
    _fseeki64(fi, this->dataOffsets[0], 0);
#else
    fseek(fi, this->dataOffsets[0], 0);
#endif

    if (dstType == DataType::FP8_E4M3) {
        this->blockN = blockN;
        this->blockM = blockM;
        this->buffer = new uint8_t[n * m];
        fread(this->buffer, 1, this->bytes, fi);
        this->scalesBuffer = new float[ns * ms];
        std::memcpy(this->scalesBuffer, scale.buffer, ns * ms * sizeof(float));
    } else {
        uint8_t *ori = new uint8_t[this->bytes];
        fread(ori, 1, this->bytes, fi);

        this->buffer = new uint8_t[n * m * sizeof(uint32_t)];
        float *floatbuffer = (float *)this->buffer;

        for (int bi = 0; bi < ns; bi++) {
            for (int bj = 0; bj < ms; bj++) {
                float curScale = ((float *)scale.buffer)[bi * ms + bj];
                for (int i = bi * blockN; i < (bi + 1) * blockN && i < n; i++) {
                    for (int j = bj * blockM; j < (bj + 1) * blockM && j < m; j++) {
                        floatbuffer[i * m + j] = curScale * fp8e4m3tofp32.dict[ori[i * m + j]];
                    }
                }
            }
        }
        delete[] ori;
    }

    fclose(fi);
}

SafeTensors::SafeTensors(const std::set<std::string> fileNames) {
    this->fileNames = fileNames;
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
        if (f[n][m]) {
            return WeightType::LINEAR;
        }
    }

    return WeightType::NONE;
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