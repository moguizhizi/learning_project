// safetensors.cpp

#include "struct_space.hpp"
#include "enum_space.h"
#include "fastllm.h"
#include "file_utils.hpp"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <numa.h>
#include <sys/mman.h>
#include <unistd.h>

const int DDRLEN = 256 * 1024 * 1024;
const int OUTPUTOFFSET = 128 * 1024 * 1024;
const int FLAGOFFSET = 255 * 1024 * 1024;
const int PAGE = 64 * 1024;
const int NUMA_PAGE = 16 * 1024;

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
    DataType srcType;
    if (this->dtype == "fastllm") {
        this->buffer = new uint8_t[this->bytes];
        fread(this->buffer, 1, this->bytes, fi);
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
        fread(this->buffer, 1, this->bytes, fi);
    } else {
        uint8_t *ori = new uint8_t[this->bytes];
        fread(ori, 1, this->bytes, fi);
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
        this->blockK = blockN;
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
                this->scalesBuffer[y * group + gx] = s;
                this->minsBuffer[y * group + gx] = -s * z;
            }
        }
        this->buffer = new uint8_t[n * m * 4];
        std::memset(this->buffer, 0, n * m * 4);
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
void WeightMap::AddTokenizerWord(const std::string &key, int value, float score) { this->tokenizer.Insert(key, value, score); }
void WeightMap::AddEmptyWeight(const std::string &key, const std::vector<int> &dims, DataType dataType) {
    Data weightData = Data(dataType, dims);
    weightData.name = key;
    this->weight[key] = weightData;
}

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

Data &WeightMap::operator[](const std::string &key) { return weight[key]; }

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

MultiThreadGroupQuantizationOp::MultiThreadGroupQuantizationOp(
    int st, int end, int m, int bit, LowBitConfig *configs, int group, int groupCnt, float *f, uint8_t *u8, int type) {
    this->st = st;
    this->end = end;
    this->bit = bit;
    this->configs = configs;
    this->group = group;
    this->groupCnt = groupCnt;
    this->f = f;
    this->u8 = u8;
    this->type = type;
}

void MultiThreadGroupQuantizationOp::Run() {

    int cid = 0, groupStart, groupEnd;
    for (int i = this->st; i < this->end; i++) {
        for (int g = 0; g < this->group; g++) {
            cid = i * group + g;
            groupStart = g * this->groupCnt;
            groupEnd = std::min((g + 1) * this->groupCnt, this->m);

            float minValue = 1e9, maxValue = -1e9;
            for (int j = groupStart; j < groupEnd; j++) {
                minValue = std::min(minValue, f[i * m + j]);
                maxValue = std::max(maxValue, f[i * m + j]);
            }
            if (this->bit == 8) {
                this->configs[cid] = LowBitConfig(maxValue, minValue, this->type, this->bit);
                for (int j = groupStart; j < groupEnd; j++) {
                    this->u8[i * m + j] = this->configs[cid].quantization(f[i * m + j]);
                }
            } else if (this->bit == 4) {
                this->configs[cid] = LowBitConfig(maxValue, minValue, this->type, this->bit);
                for (int j = groupStart; j < groupEnd; j++) {
                    uint8_t value = this->configs[cid].quantization(f[i * m + j]);
                    uint8_t id = (i * m + j) / 2;
                    if ((i * m + j) % 2) {
                        this->u8[id] = (this->u8[id] & 0xF0) | value;

                    } else {
                        this->u8[id] = (this->u8[id] & 0x0F) | value << 4;
                    }
                }
            } else if (this->bit == 2) {
                this->configs[cid] = LowBitConfig(maxValue, minValue, this->type, this->bit);
                for (int j = groupStart; j + 3 < groupEnd; j += 4) {
                    int id = (i * m + j) / 4;
                    uint8_t value0 = this->configs[cid].quantization(f[i * m + j + 0]);
                    uint8_t value1 = this->configs[cid].quantization(f[i * m + j + 1]);
                    uint8_t value2 = this->configs[cid].quantization(f[i * m + j + 2]);
                    uint8_t value3 = this->configs[cid].quantization(f[i * m + j + 3]);

                    u8[id] = value0 << 6 | value1 << 4 | value2 << 2 | value3;
                }
            }
        }
    }
}

MultiThreadGroupQuantizationBF16Op::MultiThreadGroupQuantizationBF16Op(
    int st, int end, int m, uint16_t *bf, uint8_t *u8, LowBitConfig *configs, int bit, int group, int groupCnt, int type) {
    this->st = st;
    this->end = end;
    this->m = m;
    this->bf = bf;
    this->u8 = u8;
    this->configs = configs;
    this->bit = bit;
    this->group = group;
    this->groupCnt = groupCnt;
    this->type = type;
}

void MultiThreadGroupQuantizationBF16Op::Run() {

    int cid = 0, groupStart, groupEnd;
    for (int i = this->st; i < this->end; i++) {
        for (int g = 0; g < this->group; g++) {
            cid = i * group + g;
            groupStart = g * this->groupCnt;
            groupEnd = std::min((g + 1) * this->groupCnt, this->m);

            float minValue = 1e9, maxValue = -1e9;
            for (int j = groupStart; j < groupEnd; j++) {
                minValue = std::min(minValue, g_bf16tofp32.dict[bf[i * m + j]]);
                maxValue = std::max(maxValue, g_bf16tofp32.dict[bf[i * m + j]]);
            }
            if (this->bit == 8) {
                this->configs[cid] = LowBitConfig(maxValue, minValue, this->type, this->bit);
                for (int j = groupStart; j < groupEnd; j++) {
                    this->u8[i * m + j] = this->configs[cid].quantization(g_bf16tofp32.dict[bf[i * m + j]]);
                }
            } else if (this->bit == 4) {
                this->configs[cid] = LowBitConfig(maxValue, minValue, this->type, this->bit);
                for (int j = groupStart; j < groupEnd; j++) {
                    uint8_t value = this->configs[cid].quantization(g_bf16tofp32.dict[bf[i * m + j]]);
                    uint8_t id = (i * m + j) / 2;
                    if ((i * m + j) % 2) {
                        this->u8[id] = (this->u8[id] & 0xF0) | value;

                    } else {
                        this->u8[id] = (this->u8[id] & 0x0F) | value << 4;
                    }
                }
            } else if (this->bit == 2) {
                this->configs[cid] = LowBitConfig(maxValue, minValue, this->type, this->bit);
                for (int j = groupStart; j + 3 < groupEnd; j += 4) {
                    int id = (i * m + j) / 4;
                    uint8_t value0 = this->configs[cid].quantization(g_bf16tofp32.dict[bf[i * m + j + 0]]);
                    uint8_t value1 = this->configs[cid].quantization(g_bf16tofp32.dict[bf[i * m + j + 1]]);
                    uint8_t value2 = this->configs[cid].quantization(g_bf16tofp32.dict[bf[i * m + j + 2]]);
                    uint8_t value3 = this->configs[cid].quantization(g_bf16tofp32.dict[bf[i * m + j + 4]]);

                    this->u8[id] = value0 << 6 | value1 << 4 | value2 << 2 | value3;
                }
            }
        }
    }
}

MultiThreadPerChannelQuantizationOp::MultiThreadPerChannelQuantizationOp(
    int st, int end, int m, float *f, uint8_t *u8, LowBitConfig *configs, int bit, int type) {
    this->st = st;
    this->end = end;
    this->m = m;
    this->f = f;
    this->u8 = u8;
    this->configs = configs;
    this->bit = bit;
    this->type = type;
}

void MultiThreadPerChannelQuantizationOp::Run() {
    for (int i = this->st; i < this->end; i++) {
        float minValue = 1e9, maxValue = -1e9;
        for (int j = 0; j < this->m; j++) {
            minValue = std::min(minValue, this->f[i * this->m + j]);
            maxValue = std::max(maxValue, this->f[i * this->m + j]);
        }
        if (this->bit == 8) {
            this->configs[i] = LowBitConfig(minValue, maxValue, 8, this->type);
            for (int j = 0; j < this->m; j++) {
                this->u8[i * m + j] = this->configs[i].quantization(f[i * m + j]);
            }
        } else {
            this->configs[i] = LowBitConfig(minValue, maxValue, 4, this->type);
            for (int j = 0; j < this->m; j++) {
                int id = (i * this->m + j) / 2;
                uint8_t value = this->configs[i].quantization(f[i * this->m + j]);
                if ((i * this->m + j) % 2) {
                    this->u8[id] = (u8[id] & 0xF0) | value;
                } else {
                    this->u8[id] = (u8[id] & 0xF) | (value << 4);
                }
            }
        }
    }
}

MultiThreadPerChannelQuantizationBF16Op::MultiThreadPerChannelQuantizationBF16Op(
    int st, int end, int m, uint16_t *bf, uint8_t *u8, LowBitConfig *configs, int bit, int type) {
    this->st = st;
    this->end = end;
    this->m = m;
    this->bf = bf;
    this->u8 = u8;
    this->configs = configs;
    this->bit = bit;
    this->type = type;
}

void MultiThreadPerChannelQuantizationBF16Op::Run() {
    for (int i = this->st; i < this->end; i++) {
        float minValue = 1e9, maxValue = -1e9;
        for (int j = 0; j < m; j++) {
            minValue = std::min(minValue, g_bf16tofp32.dict[bf[i * m + j]]);
            maxValue = std::max(maxValue, g_bf16tofp32.dict[bf[i * m + j]]);
        }
        if (this->bit == 8) {
            this->configs[i] = LowBitConfig(minValue, maxValue, 8, this->type);
            for (int j = 0; j < m; j++) {
                this->u8[i * m + j] = this->configs[i].quantization(g_bf16tofp32.dict[bf[i * m + j]]);
            }
        } else {
            this->configs[i] = LowBitConfig(minValue, maxValue, 4, this->type);
            for (int j = 0; j < m; j++) {
                int id = (i * m + j) / 2;
                uint8_t value = this->configs[i].quantization(g_bf16tofp32.dict[bf[i * m + j]]);
                if ((i * m + j) % 2) {
                    this->u8[id] = (this->u8[id] & 0xF0) | value;
                } else {
                    this->u8[id] = (this->u8[id] & 0xF) | (value << 4);
                }
            }
        }
    }
}

MultiThreadBase3GroupQuantizationOp::MultiThreadBase3GroupQuantizationOp(
    int st, int end, int m, float *f32, uint8_t *u8, uint16_t *halfScales, int group, int groupCnt) {
    this->st = st;
    this->end = end;
    this->m = m;
    this->f32 = f32;
    this->u8 = u8;
    this->halfScales = halfScales;
    this->group = group;
    this->groupCnt = groupCnt;
}

void MultiThreadBase3GroupQuantizationOp::Run() {
    std::vector<uint8_t> base = {1, 3, 9, 27, 81};
    int bytesPerGroup = (this->groupCnt - 1) / 5 + 1;
    for (int i = this->st; i < this->end; i++) {
        for (int g = 0; g < this->group; g++) {
            uint8_t *cur = this->u8 + i * this->group * bytesPerGroup + g * bytesPerGroup;

            int groupStart = g * this->groupCnt;
            int groupEnd = std::min((g + 1) * this->groupCnt, this->m);

            float minValue = 1e9, maxValue = -1e9, mean = 0.0;
            for (int j = groupStart; j < groupEnd; j++) {
                minValue = std::min(minValue, f32[i * m + j]);
                maxValue = std::max(maxValue, f32[i * m + j]);
                mean += fabs(f32[i * m + j]);
            }
            mean = std::max(1e-5f, mean / (groupEnd - groupStart));

            float scale = mean;
            halfScales[i * group + g] = float_to_half(scale);

            memcpy(cur, cur + bytesPerGroup, 0);
            for (int j = groupStart; j < groupEnd; j++) {
                float now = f32[i * m + j];
                uint8_t curV = (now > -scale * 0.5) + (now > scale * 0.5);
                cur[(j - groupStart) / 5] += curV * base[(j - groupStart) % 5];
            }
        }
    }
}

MultiThreadBase3GroupQuantizationBF16Op::MultiThreadBase3GroupQuantizationBF16Op(
    int st, int end, int m, uint16_t *bf, uint8_t *u8, uint16_t *halfScales, int group, int groupCnt) {
    this->st = st;
    this->end = end;
    this->m = m;
    this->bf = bf;
    this->u8 = u8;
    this->halfScales = halfScales;
    this->group = group;
    this->groupCnt = groupCnt;
}

void MultiThreadBase3GroupQuantizationBF16Op::Run() {
    std::vector<uint8_t> base = {1, 3, 9, 27, 81};
    int bytesPerGroup = ((groupCnt - 1) / 5) + 1;
    for (int i = st; i < end; i++) {
        for (int g = 0; g < group; g++) {
            uint8_t *cur = u8 + i * group * bytesPerGroup + g * bytesPerGroup;
            int cid = i * group + g;
            int groupStart = g * groupCnt;
            int groupEnd = std::min((g + 1) * groupCnt, m);

            float minValue = 1e9, maxValue = -1e9, mean = 0.0;
            for (int j = groupStart; j < groupEnd; j++) {
                minValue = std::min(minValue, g_bf16tofp32.dict[bf[i * m + j]]);
                maxValue = std::max(maxValue, g_bf16tofp32.dict[bf[i * m + j]]);
                mean += fabs(g_bf16tofp32.dict[bf[i * m + j]]);
            }
            mean = std::max(1e-5f, mean / (groupEnd - groupStart));
            float scale = mean;
            halfScales[i * group + g] = float_to_half(scale);

            memcpy(cur, cur + bytesPerGroup, 0);
            for (int j = groupStart; j < groupEnd; j++) {
                float now = g_bf16tofp32.dict[bf[i * m + j]];
                uint8_t curV = (now > -scale * 0.5) + (now > scale * 0.5);
                cur[(j - groupStart) / 5] += curV * base[(j - groupStart) % 5];
            }
        }
    }
}

ByteWriter::ByteWriter(uint8_t *cur) { this->cur = cur; }

void ByteWriter::WriteInt(int v) {
    *((int *)this->cur) = v;
    this->cur = this->cur + sizeof(int);
}

void ByteWriter::WriteFloat(float v) {
    *((float *)this->cur) = v;
    this->cur = this->cur + sizeof(float);
}

void ByteWriter::WriteString(const std::string &s) {
    WriteInt((int)s.size());
    memcpy(this->cur, s.data(), s.size());
    this->cur = this->cur + s.size();
}

void ByteWriter::WriteBytes(uint8_t *buffer, uint64_t bytes) {
    memcpy(this->cur, buffer, bytes);
    this->cur = this->cur + bytes;
}

ByteReader::ByteReader(uint8_t *data) { this->cur = data; }

int ByteReader::ReadInt() {
    int ret = *((int *)this->cur);
    this->cur = this->cur + sizeof(int);
    return ret;
}

float ByteReader::ReadFloat() {
    float ret = *((float *)this->cur);
    this->cur = this->cur + sizeof(float);
    return ret;
}

std::string ByteReader::ReadString() {
    int len = ReadInt();
    std::string ret(reinterpret_cast<const char *>(this->cur), len); // 直接从 cur 复制 len 个字节
    this->cur += len;
    return ret;
}

void ByteReader::ReadBytes(uint8_t *buffer, uint64_t bytes) {
    memcpy(buffer, this->cur, bytes);
    this->cur += bytes;
}

Tokenizer::TrieNode::TrieNode() { this->tokenId = -999999; }

Tokenizer::Tokenizer() {
    this->root = new TrieNode();
    this->specialRoot = new TrieNode();
    int n = 0;
    wchar_t special_token = L'\x0';
    for (; special_token < L'!'; special_token++, n++) {
        this->byteCharDict[L'\x100' + n] = special_token;
        this->charByteDict[special_token] = L'\x100' + n;
    }
    for (special_token = L'\x7F'; special_token < L'\xA1'; special_token++, n++) {
        this->byteCharDict[L'\x100' + n] = special_token;
        this->charByteDict[special_token] = L'\x100' + n;
    }
    this->byteCharDict[L'\x100' + n++] = L'\xAD';
    this->charByteDict[L'\xAD'] = L'\x100' + (n - 1);
}

void Tokenizer::SetTokenizerConfig(const json11::Json &config) { this->tokenizerConfig = config; }

void Tokenizer::SetChatTemplate() {
    if (!this->tokenizerConfig.is_null()) {
        if (this->tokenizerConfig["chat_template"].is_string()) {
            this->chatTemplate = this->tokenizerConfig["chat_template"].string_value();
        } else {
            this->chatTemplate = "";
        }
    } else {
        this->chatTemplate = "";
    }
}

void Tokenizer::Insert(const std::string &s, int tokenId, float score) {
    TrieNode *now = this->root;
    for (int i = 0; i < s.size(); i++) {
        if (now->next.find(s[i]) == now->next.end()) {
            now->next[s[i]] = new TrieNode();
        }
        now = now->next[s[i]];
    }

    now->tokenId = tokenId;
    now->score = score;
    this->tokenToScoreDict[tokenId] = score;
    this->tokenToStringDict[tokenId] = s;
    this->stringToTokenDict[s] = tokenId;
}

void Tokenizer::SetSpecialTokens(const std::map<std::string, int> &specialTokenMap) {
    for (const auto &it : specialTokenMap) {
        std::string specialtoken = Normalize(it.first, false);
        int tokenId = it.second;
        float score = 0.0f;
        TrieNode *now = this->specialRoot;
        for (int i = 0; i < specialtoken.size(); i++) {
            if (now->next.find(specialtoken[i]) == now->next.end()) {
                now->next[specialtoken[i]] = new TrieNode();
            }
            now = now->next[specialtoken[i]];
        }

        now->tokenId = tokenId;
        now->score = score;
        this->tokenToStringDict[tokenId] = specialtoken;
        this->stringToTokenDict[specialtoken] = tokenId;
        this->specialTokens.push_back(specialtoken);
    }
}

std::wstring Tokenizer::Utf8ToWstring(const std::string &utf8Str) {
    icu::UnicodeString unicodeStr = icu::UnicodeString::fromUTF8(utf8Str);
    std::wstring result;
    for (int i = 0; i < unicodeStr.length(); ++i) {
        result += static_cast<wchar_t>(unicodeStr.charAt(i));
    }
    return result;
}

std::string Tokenizer::WstringToUtf8(const std::wstring &wstr) {
    icu::UnicodeString unicodeStr;
    for (wchar_t wc : wstr) {
        unicodeStr.append(static_cast<UChar32>(wc));
    }
    std::string utf8Str;
    unicodeStr.toUTF8String(utf8Str);
    return utf8Str;
}

std::string Tokenizer::Normalize(const std::string &ori, const bool addDummyPrefix) {
    if (this->byteAsChar) {
        std::wstring ws(ori.size(), L' ');
        for (int i = 0; i < ori.size(); i++) {
            wchar_t wi = static_cast<wchar_t>(static_cast<unsigned char>(ori[i]));
            if (this->charByteDict.find(wi) != this->charByteDict.end()) {
                wi = this->charByteDict[wi];
            }
            ws[i] = wi;
        }
        return this->WstringToUtf8(ws);
    }

    std::string blank = "";
    blank += 226, blank += 150, blank += 129;
    std::string s = (addDummyPrefix && this->addDummyPrefix) ? blank : "";
    if (15 < ori.size() && ori.substr(0, 15) == "<FLM_FIX_TOKEN_") {
        s = "";
    }
    for (int i = 0; i < ori.size(); i++) {
        if (ori[i] == ' ') {
            if (!this->removeExtraWhitespaces && i > 0 && ori[i - 1] == ' ') {
                s += blank;
            }
        } else {
            s += ori[i];
        }
    }

    return s;
}

int Tokenizer::GetTokenId(const std::string &s) {
    AssertInFastLLM(this->stringToTokenDict.find(s) != this->stringToTokenDict.end(), "Tokenizer.GetTokenId error: can't find token \"" + s + "\"");
    return this->stringToTokenDict[s];
}

ComputeServer::ComputeServer(int partId, int partCnt, int threadNum) {
    const char *shm_name = "/fastllm_shm";
    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0x666);
    if (shm_fd == -1) {
        printf("err\n");
        exit(0);
    }

    if (ftruncate(shm_fd, DDRLEN) == -1) {
        printf("err\n");
        exit(0);
    }

    void *ptr = mmap(nullptr, DDRLEN, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        printf("err\n");
        exit(0);
    }

    char *data = static_cast<char *>(ptr);
    this->baseAddr = (volatile uint8_t *)data;
    this->baseOutputAddr = (volatile uint8_t *)(this->baseAddr + OUTPUTOFFSET);
    this->flag = (volatile int *)(baseAddr + FLAGOFFSET + partId * PAGE);

    this->inputBuffer.resize(DDRLEN);
    this->outputBuffer.resize(DDRLEN);
}

void ComputeServer::Start() {}

NumaClient::NumaClient() {
    std::string s = getenv("FASTLLM_ACTIVE_NUMA");
    if (s == "" || s == "OFF") {
        return;
    }

    std::vector<int> nodes;
    struct bitmask *mask = numa_get_mems_allowed();
    for (int i = 0; i <= numa_max_node(); i++) {
        if (numa_bitmask_isbitset(mask, i)) {
            nodes.push_back(i);
        }
    }

    int numaThreads = 27;
    try {
        std::string s = getenv("FASTLLM_NUMA_THREADS");
        if (s != "") {
            int t = atoi(s.c_str());
            if (t > 0) {
                numaThreads = t;
            }
        }
    } catch (...) {
    }

    try {
        std::string s = getenv("FASTLLM_NUMAS");
        if (s != "") {
            int t = atoi(s.c_str());
            if (t > 0 && t < nodes.size()) {
                nodes.resize(t);
            }
        }
    } catch (...) {
    }

    for (int i = 0; i < nodes.size(); i++) {
        int pid = fork();
        if (pid == 0) {
            int partId = i;
            int partCnt = nodes.size();
            int numaId = nodes[i];
            // 绑定到指定NUMA节点
            if (numa_run_on_node(numaId) != 0) {
                std::cerr << "Failed to bind process to node " << numaId << ": " << strerror(errno) << std::endl;
                exit(EXIT_FAILURE);
            }

            struct bitmask *mask = numa_bitmask_alloc(numa_num_configured_nodes());
            numa_bitmask_clearall(mask);
            numa_bitmask_setbit(mask, i);
            numa_set_membind(mask);
            numa_bitmask_free(mask);

            printf("numa server running on node %d. (part %d / %d, %d threads)\n", numaId, partId, partCnt, numaThreads);
            ComputeServer *computeServer = new ComputeServer(partId, partCnt, numaThreads);
            computeServer->Start();
        }
    }

    const char *shm_name = "/fastllm_shm";
    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0x666);
    if (shm_fd == -1) {
        printf("err\n");
        exit(0);
    }

    if (ftruncate(shm_fd, DDRLEN) == -1) {
        printf("err\n");
        exit(0);
    }

    void *ptr = mmap(nullptr, DDRLEN, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        printf("err\n");
        exit(0);
    }

    char *data = static_cast<char *>(ptr);
    this->buf = (volatile uint8_t *)data;
    this->result = this->buf + OUTPUTOFFSET;
    this->flag = (volatile int32_t *)(this->buf + FLAGOFFSET);

    this->serverNumaCnt = 4;
    this->Launch(ComputeTaskType::GetComputeServerInfo);
    while (true) {
        int a = *(this->flag);
        if (a == 0) {
            break;
        }
    }

    std::string infoString = "";
    int len = ((uint32_t *)this->result)[0];
    for (int i = 0; i < len; i++) {
        infoString += this->result[4 + i];
    }

    std::string error;
    auto info = json11::Json::parse(infoString, error);
    this->serverNumaCnt = info["numacnt"].int_value();
    this->Wait();
}

void NumaClient::Launch(int opType) {
    barrier();
    volatile int32_t *curFlag = this->flag;
    for (int i = 0; i < this->serverNumaCnt; i++) {
        *curFlag = opType;
        curFlag = curFlag + NUMA_PAGE;
        barrier();
    }
}
void NumaClient::Wait() {
    while (true) {
        int noFinish = 0;
        volatile int32_t *curFlag = this->flag;
        for (int i = 0; i < this->serverNumaCnt; i++) {
            noFinish |= *curFlag;
            curFlag = curFlag + NUMA_PAGE;
        }
        if (!noFinish) {
            return;
        }
    }
}

MultiThreadSingleAttentionOp::MultiThreadSingleAttentionOp(
    float *qd, float *kd, float *vd, float *maskd, float *od, float scale, int q1, int q2, int k1, int v2) {
    this->q1 = q1;
    this->q2 = q2;
    this->k1 = k1;
    this->v2 = v2;
    this->od = od;
    this->qd = qd;
    this->kd = kd;
    this->vd = vd;
    this->maskd = maskd;
    this->scale = scale;
}

void MultiThreadSingleAttentionOp::Run() {

    int base = k1 - q1;
    for (int i = 0; i < this->q1; i++) {

        float *qk = new float[this->k1]();
        float *temp = new float[this->k1]();

        float maxValue = -10000, sum = 0.0;
        for (int j = 0; j < this->k1; j++) {
            if (maskd && maskd[i * k1 + j] > 0.99) {
                qk[j] = -10000;
                continue;
            }
            if (!maskd && (base + i) < j) {
                qk[j] = -10000;
                continue;
            }

            int l = 0;
            float now = 0.0f;
#ifdef __aarch64__
            float32x4_t sum = {0, 0, 0, 0};
            for (; l + 3 < q2; l += 4) {
                sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(qd + i * q2 + l), vld1q_f32(kd + j * q2 + l)));
            }
            now += sum[0] + sum[1] + sum[2] + sum[3];
#elif defined(__AVX__)
            __m256 vsum = _mm256_set1_ps(0.0f);
            for (; l + 7 < q2; l += 8) {
                __m256 vx = _mm256_loadu_ps((const float *)(qd + i * q2 + l));
                __m256 vy = _mm256_loadu_ps((const float *)(kd + j * q2 + l));
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vy));
            }
            now += Floatsum(vsum);
#endif
            for (; l < this->q2; l++) {
                now = now + this->qd[i * this->q2 + l] * this->kd[j * this->q2 + l];
            }
            qk[j] = now * this->scale;
            maxValue = std::max(qk[j], maxValue);
        }

        int j = 0;
#ifdef __aarch64__
        float32x4_t vmax = vdupq_n_f32(maxValue);
        for (; j + 3 < k1; j += 4) {
            vst1q_f32(temp + j, exp_ps(vsubq_f32(vld1q_f32(qk + j), vmax)));
        }
#endif

        for (; j < this->k1; j++) {
            temp[j] = expf(qk[j] - maxValue);
        }

        sum = 0.0f;
        for (int j = 0; j < k1; j++) {
            sum += temp[j];
        }
        sum = std::max(sum, 0.1f);

        for (int j = 0; j < this->k1; j++) {
            qk[j] = qk[j] / sum;
        }

        for (int j = 0; j < this->k1; j++) {
            for (int l = 0; l < this->v2; l++) {
                this->od[i * this->v2 + l] += qk[j] * this->vd[j * this->v2 + l];
            }
        }

        delete[] qk;
        delete[] temp;
    }
}

MultiThreadSingleAttentionFloat16Op::MultiThreadSingleAttentionFloat16Op(
    uint16_t *qd, uint16_t *kd, uint16_t *vd, uint16_t *maskd, uint16_t *od, float scale, int q1, int q2, int k1, int v2) {

    this->qd = qd;
    this->kd = kd;
    this->vd = vd;
    this->maskd = maskd;
    this->od = od;
    this->scale = scale;
    this->q1 = q1;
    this->q2 = q2;
    this->k1 = k1;
    this->v2 = v2;
}

void MultiThreadSingleAttentionFloat16Op::Run() {
    std::vector<float> fqd, fkd, fvd, fod, fmaskd;

    fqd.resize(this->q1 * this->q2);
    fkd.resize(this->k1 * this->q2);
    fvd.resize(this->k1 * this->v2);
    fmaskd.resize(this->maskd ? this->q1 * this->k1 : 0);
    fod.resize(this->q1 * this->v2);
    if (this->maskd) {
        Float16ToFloat32(this->maskd, fmaskd.data(), (int)fmaskd.size());
    }

    Float16ToFloat32(this->qd, fqd.data(), (int)fqd.size());
    Float16ToFloat32(this->kd, fkd.data(), (int)fkd.size());
    Float16ToFloat32(this->vd, fvd.data(), (int)fvd.size());
    Float16ToFloat32(this->od, fod.data(), (int)fod.size());

    MultiThreadSingleAttentionOp(
        fqd.data(), fkd.data(), fvd.data(), this->maskd ? fmaskd.data() : nullptr, fod.data(), this->scale, this->q1, this->q2, this->k1, this->v2)
        .Run();

    Float32ToFloat16(fod.data(), this->od, (int)fod.size());
}