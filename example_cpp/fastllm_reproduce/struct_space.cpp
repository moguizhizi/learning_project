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

void SafeTensorItem::CreateBufferWithAWQ(DataType dstType, SafeTensorItem &scale, SafeTensorItem &qzero) {
    const int groupCnt = this->intShape[0] / qzero.intShape[0];

    AssertInFastLLM(this->shape.size() == 2 && scale.shape.size() == 2 && qzero.shape.size() == 2,
                    "CreateBufferWithAWQ error: shape.size() should be 2.");
    AssertInFastLLM(groupCnt * scale.shape[0] == this->shape[0] && groupCnt * qzero.shape[0] == this->shape[0] &&
                        8 * this->shape[1] == scale.shape[1] && this->shape[1] == qzero.shape[1],
                    "CreateBufferWithAWQ error: shape error.");
    AssertInFastLLM(this->dtype == "I32" && qzero.dtype == "I32", "CreateBufferWithAWQ error: dtype shoud be I32.");

    this->ClearBuffer();

    FILE *fweight = fopen(this->fileName.c_str(), "rb");
    FILE *fqzero = fopen(qzero.fileName.c_str(), "rb");

#if defined(_WIN32) || defined(_WIN64)
    _fseeki64(fqweight, this->dataOffsets[0], 0);
    _fseeki64(fqzero, qzero.dataOffsets[0], 0);
#else
    fseek(fweight, this->dataOffsets[0], 0);
    fseek(fqzero, qzero.dataOffsets[0], 0);
#endif

    uint8_t *ori_weight = new uint8_t[this->bytes];
    uint8_t *ori_qzero = new uint8_t[qzero.bytes];

    int ret;
    ret = fread(ori_weight, 1, this->bytes, fweight);
    ret = fread(ori_qzero, 1, qzero.bytes, fqzero);

    int n = this->intShape[0];
    int m = this->intShape[1];

    unsigned int *weight_int32 = (unsigned int *)ori_weight;
    unsigned int *qzero_int32 = (unsigned int *)ori_qzero;
    float *scale_f32 = (float *)scale.buffer;

    static const int awq_shift[8] = {0, 16, 4, 20, 8, 24, 12, 28}; // awq order = [0,2,4,8,1,3,5,7]

    if (dstType == DataType::FLOAT32) {
        this->buffer = new uint8_t[n * m * 8 * 4];
        float *floatBuffer = (float *)this->buffer;
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < m * 8; y++) {
                int gx = x / groupCnt;
                int gy = y >> 3;
                unsigned int w = (weight_int32[x * m + gy] >> awq_shift[y & 7]) & 15;
                unsigned int z = (qzero_int32[gx * m + gy] >> awq_shift[y & 7]) & 15;
                float s = scale_f32[gx * m * 8 + y];
                floatBuffer[y * n + x] = (w - z) * s;
            }
        }
    } else if (dstType == DataType::INT4_GROUP) {
        int group = (n - 1) / groupCnt + 1;
        this->scalesBuffer = new float[m * 8 * group];
        this->minsBuffer = new float[m * 8 * group];
        for (int x = 0; x < n; x += groupCnt) {
            for (int y = 0; y < m * 8; y++) {
                int gx = x / groupCnt;
                int gy = y >> 3;
                unsigned int z = (qzero_int32[gx * m + gy] >> awq_shift[y & 7]) & 15;
                float s = scale_f32[gx * m * 8 + y];
                this->scalesBuffer[y * group + x] = s;
                this->minsBuffer[y * group + x] = -z * s;
            }
        }
        this->buffer = new uint8_t[n * m * 8 * 0.5];
        std::memset(this->buffer, 0, n * m * 8 * 0.5);
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < m * 8; y++) {
                int gy = y >> 3;
                int w = (weight_int32[x * m + gy] >> awq_shift[y & 7]) & 15;
                buffer[y * n / 2 + x / 2] += (w << ((1 - (x & 1)) * 4));
            }
        }

    } else {
        ErrorInFastLLM("CreateBufferWithAWQ Error: dst type error.");
    }

    fclose(fweight);
    fclose(fqzero);
    delete[] ori_weight;
    delete[] ori_qzero;
}

void SafeTensorItem::Transpose(DataType type) {
    int n = intShape[0], m = intShape[1];
    if (type == DataType::FLOAT32) {
        float *temp = new float[len];
        memcpy(temp, this->buffer, len * sizeof(float));
        TransposeF32((float *)this->buffer, temp, n, m, n, m);
        delete[] temp;
    } else if (type == DataType::FLOAT16 || type == DataType::BFLOAT16) {
        uint16_t *temp = new uint16_t[len];
        memcpy(temp, this->buffer, len * sizeof(uint16_t));
        TransposeSimple((uint16_t *)this->buffer, temp, n, m, n, m);
        delete[] temp;
    } else {
        ErrorInFastLLM("SafeTensorItem.Transpose: unsupport dtype " + std::to_string(type) + "\n");
    }
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