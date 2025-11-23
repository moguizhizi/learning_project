// safetensors.hpp

#pragma once

#include <cpuid.h>
#include <immintrin.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "common_class.h"
#include "common_struct.h"
#include "enum_space.h"
#include "fastllm-cuda.cuh"
#include "json11.hpp"

struct SafeTensorItem {
    std::string tensorName;
    std::string fileName;
    std::string dtype;
    std::vector<uint64_t> shape;
    std::vector<int> intShape;
    std::vector<uint64_t> dataOffsets;

    uint64_t len = 1;
    uint64_t bytes = 1;
    int blockK;
    int blockM;

    uint8_t *buffer = nullptr;
    float *minsBuffer = nullptr, *scalesBuffer = nullptr;

    SafeTensorItem();
    ~SafeTensorItem();
    SafeTensorItem(const std::string &tensorName, const std::string &fileName, uint64_t baseOffset, const json11::Json &config);
    void ClearBuffer();
    void CreateBuffer(DataType dstType);
    void CreateBufferWithScale(DataType dstType, SafeTensorItem &scale);
    void CreateBufferWithAWQ(DataType dstType, SafeTensorItem &scale, SafeTensorItem &qzero);
    void Transpose(DataType type);
};

struct SafeTensors {
    std::map<std::string, SafeTensorItem> itmeDict;
    std::set<std::string> fileNames;

    SafeTensors(const std::set<std::string> fileNames);

    std::vector<std::string> GetSortedItemNames();
};

struct Tokenizer {
    enum TokenizerType { BPE = 0, NORMAL = 1, QWEN = 2, GLM = 3, BERT = 4 };

    struct TrieNode {
        int tokenId;
        float score;
        std::map<int, TrieNode *> next;
        TrieNode();
    };

    json11::Json tokenizerConfig;
    std::string chatTemplate = "";

    bool addDummyPrefix = true;         // 是否在首位添加空格
    bool removeExtraWhitespaces = true; // 是否将多个空格合并为一个
    bool byteAsChar = false;            // 是否将byte变为展示字符

    std::unordered_map<int, std::string> tokenToStringDict;
    std::unordered_map<int, float> tokenToScoreDict;
    std::unordered_map<std::string, int> stringToTokenDict;
    std::vector<std::string> specialTokens;

    std::unordered_map<wchar_t, wchar_t> byteCharDict;
    std::unordered_map<wchar_t, wchar_t> charByteDict;

    TrieNode *root = nullptr;
    TrieNode *specialRoot = nullptr;

    TokenizerType type = TokenizerType::BPE;

    Tokenizer();
    void SetTokenizerConfig(const json11::Json &config);
    void SetChatTemplate();
    void Insert(const std::string &s, int tokenId, float score = 1.0f); // 插入一个token
    std::string Normalize(const std::string &ori, const bool addDummyPrefix = true);
    void SetSpecialTokens(const std::map<std::string, int> &specialTokenMap);
    std::string WstringToUtf8(const std::wstring &wstr);
    std::wstring Utf8ToWstring(const std::string &utf8Str);
    int GetTokenId(const std::string &s);
};

struct WeightMap {
    Tokenizer tokenizer;

    std::set<std::string> embeddingsNames;
    std::set<std::string> linearNames;
    std::map<std::string, std::string> dicts;
    std::map<std::string, Data> weight;

    void AddDict(const std::string &key, const std::string &value);
    void AddTokenizerWord(const std::string &key, int value, float score);
    void AddEmptyWeight(const std::string &key, const std::vector<int> &dims, DataType dataType);
    WeightType GetWeightType(const std::string &key);
    Data &operator[](const std::string &key);
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

struct MultiThreadBaseOp {
    virtual void Run() = 0;
};

struct MultiThreadGroupQuantizationOp : MultiThreadBaseOp {
    int st, end, m;
    float *f;
    uint8_t *u8;
    LowBitConfig *configs;
    int bit;
    int group, groupCnt;
    int type;

    MultiThreadGroupQuantizationOp(
        int st, int end, int m, int bit, LowBitConfig *configs, int group, int groupCnt, float *f, uint8_t *u8, int type);
    void Run() override;
};

struct MultiThreadGroupQuantizationBF16Op : MultiThreadBaseOp {
    int st, end, m;
    uint16_t *bf;
    uint8_t *u8;
    LowBitConfig *configs;
    int bit;
    int group, groupCnt;
    int type;

    MultiThreadGroupQuantizationBF16Op(
        int st, int end, int m, uint16_t *bf, uint8_t *u8, LowBitConfig *configs, int bit, int group, int groupCnt, int type);
    void Run() override;
};

struct MultiThreadPerChannelQuantizationOp : MultiThreadBaseOp {
    int st, end, m;
    float *f;
    uint8_t *u8;
    LowBitConfig *configs;
    int bit;
    int type;

    MultiThreadPerChannelQuantizationOp(int st, int end, int m, float *f, uint8_t *u8, LowBitConfig *configs, int bit, int type);
    void Run() override;
};

struct MultiThreadPerChannelQuantizationBF16Op : MultiThreadBaseOp {
    int st, end, m;
    uint16_t *bf;
    uint8_t *u8;
    LowBitConfig *configs;
    int bit;
    int type;

    MultiThreadPerChannelQuantizationBF16Op(int st, int end, int m, uint16_t *bf, uint8_t *u8, LowBitConfig *configs, int bit, int type);
    void Run() override;
};

struct MultiThreadBase3GroupQuantizationOp : MultiThreadBaseOp {
    int st, end, m;
    float *f32;
    uint8_t *u8;
    uint16_t *halfScales;
    int group;
    int groupCnt;

    MultiThreadBase3GroupQuantizationOp(int st, int end, int m, float *f32, uint8_t *u8, uint16_t *scale, int group, int groupCnt);
    void Run() override;
};

struct MultiThreadBase3GroupQuantizationBF16Op : MultiThreadBaseOp {
    int st, end, m;
    uint16_t *bf;
    uint8_t *u8;
    uint16_t *halfScales;
    int group, groupCnt;

    MultiThreadBase3GroupQuantizationBF16Op(int st, int end, int m, uint16_t *bf, uint8_t *u8, uint16_t *halfScales, int group, int groupCnt);
    void Run() override;
};

struct ByteWriter {
    uint8_t *cur;

    ByteWriter(uint8_t *data);

    void WriteInt(int v);
    void WriteFloat(float v);
    void WriteString(const std::string &s);
    void WriteBytes(uint8_t *buffer, uint64_t bytes);
};

struct ByteReader {
    uint8_t *cur;

    ByteReader(uint8_t *data);
    int ReadInt();
    float ReadFloat();
    std::string ReadString();
    void ReadBytes(uint8_t *buffer, uint64_t bytes);
};

struct ComputeServer {
    volatile uint8_t *baseAddr;
    volatile uint8_t *baseOutputAddr;
    volatile int *flag;

    std::vector<uint8_t> inputBuffer;
    std::vector<uint8_t> outputBuffer;

    ComputeServer(int partId, int partCnt, int threadNum);
    void Start();
};

struct NumaClient {
    int serverNumaCnt;

    volatile int32_t *flag;
    volatile uint8_t *result;
    volatile uint8_t *buf;

    NumaClient();

    void Launch(int opType);
    void Wait();
};

struct MultiThreadSingleAttentionOp : MultiThreadBaseOp {
    float *qd, *kd, *vd, *maskd, *od;
    float scale;
    int q1, q2, k1, v2;

    MultiThreadSingleAttentionOp(float *qd, float *kd, float *vd, float *maskd, float *od, float scale, int q1, int q2, int k1, int v2);
    void Run();
};

struct MultiThreadSingleAttentionFloat16Op : MultiThreadBaseOp {
    uint16_t *qd, *kd, *vd, *maskd, *od;
    float scale;
    int q1, q2, k1, v2;
    MultiThreadSingleAttentionFloat16Op(
        uint16_t *qd, uint16_t *kd, uint16_t *vd, uint16_t *maskd, uint16_t *od, float scale, int q1, int q2, int k1, int v2);

    void Run();
};

struct MultiThreadRMSNormFloatOp : MultiThreadBaseOp {
    float *output, *input, *weight;
    int outer, channels;
    float eps;

    MultiThreadRMSNormFloatOp(float *output, float *input, float *weight, int outer, int channels, float eps);
    void Run();
};

struct MultiThreadInt4GroupLinearOp : MultiThreadBaseOp {
    float *inputData;
    uint8_t *weightData;
    float *biasData, *outputData;
    uint16_t *mins, *scales;
    int n, m, k, st, end, group, groupCnt;

    MultiThreadInt4GroupLinearOp(float *inputData, uint8_t *weightData, float *biasData, float *outputData, uint16_t *mins, uint16_t *scales,
        int n, int m, int k, int st, int end, int group, int groupCnt);

    void Run();
};

struct MultiThreadBase3GroupLinearOp : MultiThreadBaseOp {
    float *inputData;
    uint8_t *weightData;
    float *biasData, *outputData;
    int n, m, k, st, end, group, groupCnt;
    uint16_t *halfScales;

    MultiThreadBase3GroupLinearOp(float *inputData, uint8_t *weightData, float *biasData, float *outputData, int n, int m, int k, int st,
        int end, int group, int groupCnt, uint16_t *halfScales);

    void Run();
};

struct MultiThreadFloat32ToBFloat16Op : MultiThreadBaseOp {
    float *input;
    uint16_t *output;
    int len;

    MultiThreadFloat32ToBFloat16Op(float *input, uint16_t *output, int len);
    void Run();
};

struct MultiCudaDoLinearOp : MultiThreadBaseOp {
    uint8_t *oriCudaInput, *oriCpuInput;
    Data *input, *weight, *bias;
    Data *output;
    int n, m, k, start, len;
    uint8_t *lastOutput;
    int deviceId;

    MultiCudaDoLinearOp(uint8_t *oriCudaInput, uint8_t *oriCpuInput, Data *input, Data *weight, Data *bias, Data *output, int n, int m, int k,
        int start, int len, uint8_t *lastOutput, int deviceId);

    void Run();
};

struct MultiCudaDoMergeAttentionOp : MultiThreadBaseOp {
    uint8_t *oriCudaInput, *oriCpuInput, *partOutput;
    Data *input, *weight0, *bias0, *weight1, *bias1;
    Data *qkv, *q, *k, *v;
    int qNum, kvNum, headDim, rotDim;
    float attentionScale;
    Data *positionIds, *sinData, *cosData;
    Data **keys, **values, **masks;
    Data *output;
    int batch;
    int deviceId;

    MultiCudaDoMergeAttentionOp(uint8_t *oriCudaInput, uint8_t *oriCpuInput, uint8_t *partOutput, Data *input, Data *weight0, Data *bias0,
        Data *weight1, Data *bias1, Data *qkv, Data *q, Data *k, Data *v, int qNum, int kvNum, int headDim, int rotDim, float attentionScale,
        Data *positionIds, Data *sinData, Data *cosData, Data **keys, Data **values, Data **masks, Data *output, int batch, int deviceId);

    void Run();
};

struct MultiCudaDoMergeMLPOp : MultiThreadBaseOp {
    uint8_t *oriCudaInput, *oriCpuInput, *partOutput;
    Data *input, *weight0, *bias0, *weight1, *bias1;
    Data *w1, *w2, *w3;
    Data *output;
    int deviceId;

    MultiCudaDoMergeMLPOp(uint8_t *oriCudaInput, uint8_t *oriCpuInput, uint8_t *partOutput, Data *input, Data *weight0, Data *bias0,
        Data *weight1, Data *bias1, Data *w1, Data *w2, Data *w3, Data *output, int deviceId);

    void Run();
};

struct MultiCudaCpuDoMergeMLPOp : MultiThreadBaseOp {
    uint8_t *oriCpuInput, *partOutput;
    Data *input, *weight0, *bias0, *weight1, *bias1;
    Data *w1, *w2, *w3;
    Data *output;
    int deviceId;

    MultiCudaCpuDoMergeMLPOp(uint8_t *oriCpuInput, uint8_t *partOutput, Data *input, Data *weight0, Data *bias0, Data *weight1, Data *bias1,
        Data *w1, Data *w2, Data *w3, Data *output, int deviceId);

    void Run();
};

struct ExpertRoute {
    int expertIndex;
    float weight;
};

struct MultiCudaDoMergeMOEOp : MultiThreadBaseOp {
    uint8_t *oriCudaInput, *oriCpuInput, *partOutput;
    Data *input;
    Data **weights;
    Data *logits, *gateBias;
    Data *w1, *w2, *w3;
    int wBatch, topk, needNorm;
    float routeScale, sharedScale;
    Data *output;
    int deviceId;
    std::vector<Data *> deviceWeights;

    MultiCudaDoMergeMOEOp(uint8_t *oriCudaInput, uint8_t *oriCpuInput, uint8_t *partOutput, Data *input, Data **weights, Data *logits,
        Data *gateBias, Data *w1, Data *w2, Data *w3, int wBatch, int topk, int needNorm, float routeScale, float sharedScale, Data *output,
        int deviceId);

    void PrepareInputBuffer();
    void MapWeightsForDevice();
    void ComputeMoE();
    void FinalizeOutputBuffer();
    void Run();
};

struct MultiCudaCpuDoMergeMOEOp : MultiThreadBaseOp {
    uint8_t *oriCpuInput, *partOutput;
    Data *input;
    Data **weights;
    Data *logits, *gateBias;
    Data *w1, *w2, *w3;
    int wBatch, topk, needNorm;
    float routeScale, sharedScale;
    Data *output;
    int deviceId;

    MultiCudaCpuDoMergeMOEOp(uint8_t *oriCpuInput, uint8_t *partOutput, Data *input, Data **weights, Data *logits, Data *gateBias, Data *w1,
        Data *w2, Data *w3, int wBatch, int topk, int needNorm, float routeScale, float sharedScale, Data *output, int deviceId);

    void Run();
};

struct MultiThreadMultiOps : MultiThreadBaseOp {
    std::vector<MultiThreadBaseOp *> ops;

    void Run();

    ~MultiThreadMultiOps();
};

struct MultiThreadLinearBFloat16FP8E4M3Op : MultiThreadBaseOp {
    uint16_t *inputData;
    uint8_t *weightData;
    float *biasData, *outputData;
    int n, m, k, st, end;
    int blockK, blockM;
    float *scales;

    MultiThreadLinearBFloat16FP8E4M3Op(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData, int n, int m, int k, int st,
        int end, float *scales, int blockK, int blockM);

    void Run();
};

struct MultiThreadLinearInt8Int4GroupOp : MultiThreadBaseOp {
    uint8_t *a, *b;
    float *c;
    int n, m, k, kstride;
    int *weightSums;
    float *weightMins;
    float *scales;
    float *bias;
    float *iscales, *izeros;
    float *inputSums;
    int group, groupCnt;

    MultiThreadLinearInt8Int4GroupOp(uint8_t *a, uint8_t *b, float *c, int n, int m, int k, int kstride, int *weightSums, float *weightMins,
        float *scales, float *bias, float *iscales, float *izeros, float *inputSums, int group, int groupCnt);

    void Run();
};

struct MultiThreadLinearFloat32Float32Op : MultiThreadBaseOp {
    float *inputData;
    float *weightData;
    float *biasData, *outputData;
    int n, m, k, st, end;

    MultiThreadLinearFloat32Float32Op(
        float *inputData, float *weightData, float *biasData, float *outputData, int n, int m, int k, int st, int end);

    void Run();
};

struct MultiThreadLinearFloat32Float16Op : MultiThreadBaseOp {
    float *inputData;
    uint16_t *weightData;
    float *biasData, *outputData;
    int n, m, k, st, end;

    MultiThreadLinearFloat32Float16Op(
        float *inputData, uint16_t *weightData, float *biasData, float *outputData, int n, int m, int k, int st, int end);

    void Run();
};

struct CPUInstructInfo {
    bool hasAVX512F = false;
    bool hasAVX512BF16 = false;
    bool hasAVX512VNNI = false;
    // You could add more, e.g., hasAVX, hasAVX2
    CPUInstructInfo() {
#ifndef __aarch64__
#    if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
        std::array<int, 4> regs; // For EAX, EBX, ECX, EDX
        // Step 1: Check OSXSAVE bit (CPUID EAX=1, ECX bit 27)
        // This indicates if the OS supports XGETBV to query enabled AVX features
        bool os_supports_xsave = false;
#        if defined(_MSC_VER)
        __cpuid(regs.data(), 1);
#        else // GCC/Clang
        __get_cpuid(1, (unsigned int *)&regs[0], (unsigned int *)&regs[1], (unsigned int *)&regs[2], (unsigned int *)&regs[3]);
#        endif
        if (regs[2] & (1 << 27)) { // Check ECX bit 27 (OSXSAVE)
            os_supports_xsave = true;
        }
        bool os_avx_enabled = false;
        if (os_supports_xsave) {
            // Step 2: Check if AVX states (and by extension AVX512 states) are enabled by OS
            // XCR0 register:
            // Bit 1 (SSE state) must be 1
            // Bit 2 (AVX state - YMM registers) must be 1
            // Bits 5,6,7 (AVX512 OPMASK, ZMM_Hi256, Hi16_ZMM states) must be 1 for AVX512
            // We check for mask 0xE6 (binary 11100110) which means SSE, AVX, and AVX512 states are enabled
            uint64_t xcr0 = _xgetbv(0); // _XCR_XFEATURE_ENABLED_MASK is typically 0
            if ((xcr0 & 0xE6) == 0xE6) {
                os_avx_enabled = true;
            }
        }
        if (os_avx_enabled) {
// CPUID with EAX=7, ECX=0 for extended features
#        if defined(_MSC_VER)
            __cpuidex(regs.data(), 7, 0);
#        else // GCC/Clang
            __get_cpuid_count(7, 0, (unsigned int *)&regs[0], (unsigned int *)&regs[1], (unsigned int *)&regs[2], (unsigned int *)&regs[3]);
#        endif
            // AVX512F: EAX=7, ECX=0, EBX bit 16
            hasAVX512F = (regs[1] & (1 << 16)) != 0;
            // AVX512VNNI: EAX=7, ECX=0, ECX bit 11
            hasAVX512VNNI = (regs[2] & (1 << 11)) != 0;
// AVX512_BF16: EAX=7, ECX=1, EAX bit 5
// Need to make another CPUID call with ECX=1
#        if defined(_MSC_VER)
            __cpuidex(regs.data(), 7, 1);
#        else // GCC/Clang
            __get_cpuid_count(7, 1, (unsigned int *)&regs[0], (unsigned int *)&regs[1], (unsigned int *)&regs[2], (unsigned int *)&regs[3]);
#        endif
            hasAVX512BF16 = (regs[0] & (1 << 5)) != 0;
            // Important: If a feature (like AVX512_BF16) depends on another (like AVX512F),
            // you might want to ensure the base feature is also true.
            // e.g., hasAVX512BF16 = hasAVX512BF16 && hasAVX512F; (Though CPUID should report correctly)
        }
// If os_avx_enabled is false, all 'has...' flags will remain false.
#    endif // Compiler check
        // Print the results
        std::string x[2] = {"OFF", "ON"};
        printf("CPU Instruction Info: ");
        printf("[AVX512F: %s] ", x[hasAVX512F].c_str());
        printf("[AVX512_VNNI: %s] ", x[hasAVX512VNNI].c_str());
        printf("[AVX512_BF16: %s] ", x[hasAVX512BF16].c_str());
        printf("\n");
#endif // ifndef __aarch64__
    }
};