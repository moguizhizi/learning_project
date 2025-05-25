#include <sstream>
#include <fstream>
#include <set>
#include <string>
#include <iostream>
#include <map>
#include "json11.hpp"
#include <algorithm>
#include "utils.h"
#include <cstring>

enum DataType
{
    FLOAT32 = 0,
    BFLOAT16 = 1,
    INT16 = 2,
    INT8 = 3,
    INT4 = 4,
    INT2 = 5,
    BIT = 6,
    FLOAT16 = 7,
    INT4_NOZERO = 8, // 不用zeroPoint的int4, floatValue = min + uint4Value * scale
    INT4_GROUP = 9,  // 不用zeroPoint的int4, floatValue = min + uint4Value * scale, 且使用分组量化
    FP8_E4M3 = 10,
    INT2_GROUP = 11,  // 不用zeroPoint的int2, floatValue = min + uint4Value * scale, 且使用分组量化
    BASE3_GROUP = 12, // 三元量化，-1 0 1
    INT32PARAM = 100, // int32的参数，这种类型的数据永远存在CPU上
    DATA_AUTO_NONE = 99999,
    DATA_AUTO_LINEAR,
    DATA_AUTO_EMBEDDING,
    DATA_AUTO_CONV
};

static void ErrorInFastLLM_A(const std::string &error)
{
    printf("FastLLM Error: %s\n", error.c_str());
    throw error;
}

static void AssertInFastLLM(bool condition, const std::string &error)
{
    if (!condition)
    {
        ErrorInFastLLM_A(error);
    }
}

void Transpose4x4(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m)
{
    if (n < 4 || m < 4)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                pDst[j * dstStride + i] = pSrc[i * srcStride + j];
            }
        }

        return;
    }
}

void Transpose_A(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m)
{
    int per = 4;
    for (int i = 0; i < n; i += per)
    {
        for (int j = 0; j < m; j += per)
        {
            Transpose4x4(pDst + j * dstStride + i,
                         pSrc + i * srcStride + j,
                         dstStride, srcStride,
                         std::min(per, n - i),
                         std::min(per, m - j));
        }
    }
}

template <typename T>
void TransposeSimple(T *pDst, T *pSrc, int dstStride, int srcStride, int n, int m)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            pDst[j * dstStride + i] = pSrc[i * srcStride + j];
        }
    }
}

void ConvertDataType(uint8_t *src, DataType srcDtype, uint8_t *dst, DataType dstDtype, uint64_t len)
{
    if (srcDtype == dstDtype)
    {
        int unitSize = 4;
        if (dstDtype == DataType::FLOAT32)
        {
            unitSize = 4;
        }
        else if (dstDtype == DataType::FLOAT16 || dstDtype == DataType::BFLOAT16)
        {
            unitSize = 2;
        }
        else
        {
            ErrorInFastLLM_A("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
        }
        memcpy(dst, src, len * unitSize);
    }
    else if (srcDtype == DataType::FP8_E4M3 && dstDtype == DataType::FLOAT16)
    {
        ErrorInFastLLM_A("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
    }
    else if (srcDtype == DataType::BFLOAT16 && dstDtype == DataType::FLOAT32)
    {
        uint16_t *u16dst = (uint16_t *)dst;
        uint16_t *u16src = (uint16_t *)src;
        for (size_t i = 0; i < len; i++)
        {
            u16dst[i * 2] = 0;
            u16dst[i * 2 + 1] = u16src[i];
        }
    }
    else if (srcDtype == DataType::FLOAT16 && dstDtype == DataType::FLOAT32)
    {
        float *fdst = (float *)dst;
        uint16_t *u16src = (uint16_t *)src;
        for (size_t i = 0; i < len; i++)
        {
            fdst[i] = fastllm::half_to_float(u16src[i]);
        }
    }
    else
    {
        ErrorInFastLLM_A("ConvertDataType Failed. (" + std::to_string(srcDtype) + " -> " + std::to_string(dstDtype) + ")");
    }
}

struct FP8E4M3ToFP32Manager
{
    float dict[256] = {
        0.0, 0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625, 0.01171875, 0.013671875, 0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375, 0.029296875, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375, 0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480, -0.0, -0.001953125, -0.00390625, -0.005859375, -0.0078125, -0.009765625, -0.01171875, -0.013671875, -0.015625, -0.017578125, -0.01953125, -0.021484375, -0.0234375, -0.025390625, -0.02734375, -0.029296875, -0.03125, -0.03515625, -0.0390625, -0.04296875, -0.046875, -0.05078125, -0.0546875, -0.05859375, -0.0625, -0.0703125, -0.078125, -0.0859375, -0.09375, -0.1015625, -0.109375, -0.1171875, -0.125, -0.140625, -0.15625, -0.171875, -0.1875, -0.203125, -0.21875, -0.234375, -0.25, -0.28125, -0.3125, -0.34375, -0.375, -0.40625, -0.4375, -0.46875, -0.5, -0.5625, -0.625, -0.6875, -0.75, -0.8125, -0.875, -0.9375, -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875, -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0, -18.0, -20.0, -22.0, -24.0, -26.0, -28.0, -30.0, -32.0, -36.0, -40.0, -44.0, -48.0, -52.0, -56.0, -60.0, -64.0, -72.0, -80.0, -88.0, -96.0, -104.0, -112.0, -120.0, -128.0, -144.0, -160.0, -176.0, -192.0, -208.0, -224.0, -240.0, -256.0, -288.0, -320.0, -352.0, -384.0, -416.0, -448.0, -480};
};

struct SafeTensorItem
{
    std::string tensorName;
    std::string fileName;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::vector<int> intShape;
    std::vector<std::uint64_t> data_offsets;

    uint64_t len, bytes;
    uint8_t *buffer = nullptr;
    float *minsBuffer = nullptr, *scalesBuffer = nullptr;
    int blockK, blockM;

    SafeTensorItem() {}

    ~SafeTensorItem()
    {
        ClearBuffer();
    }

    SafeTensorItem(const std::string &tensorName, const std::string &fileName, const json11::Json &config, uint64_t baseOffset)
    {
        this->tensorName = tensorName;
        this->fileName = fileName;

        this->dtype = config["dtype"].string_value();
        for (auto &it : config["data_offsets"].array_items())
        {
            this->data_offsets.push_back(baseOffset + it.ll_value());
        }
        for (auto &it : config["shape"].array_items())
        {
            this->shape.push_back(it.ll_value());
            this->intShape.push_back(this->shape.back());
        }

        len = 1;
        for (auto &it : shape)
        {
            len *= it;
        }
        bytes = this->data_offsets[1] - this->data_offsets[0];
    }

    struct FP8E4M3ToFP32Manager fp8e4m3tofp32;

    void CreateBufferWithScale(DataType dstType, SafeTensorItem &scale)
    {
        AssertInFastLLM(this->shape.size() == 2 && scale.shape.size() == 2, "CreateBufferWithScale error: shape.size() should be 2.");
        DataType srcType;
        if (this->dtype == "F8_E4M3")
        {
            srcType = DataType::FP8_E4M3;
        }
        else
        {
            // ErrorInFastLLM_A("CreateBufferWithScale error: dtype should be FP8_E4M3");
        }
        int n = this->shape[0], m = this->shape[1];
        int ns = scale.shape[0], ms = scale.shape[1];
        int blockN = n / ns, blockM = m / ms;

        while ((blockN & -blockN) != blockN)
        {
            blockN++;
        }
        while ((blockM & -blockM) != blockM)
        {
            blockM++;
        }
        ClearBuffer();

        if (dstType == DataType::FP8_E4M3)
        {
            this->blockK = blockN;
            this->blockM = blockM;
            buffer = new uint8_t[n * m];
            FILE *fi = fopen(this->fileName.c_str(), "rb");
#if defined(_WIN32) || defined(_WIN64)
            _fseeki64(fi, this->data_offsets[0], 0);
#else
            fseek(fi, this->data_offsets[0], 0);
#endif
            int ret = fread(buffer, 1, this->bytes, fi);
            fclose(fi);

            scalesBuffer = new float[ns * ms];
            memcpy(scalesBuffer, scale.buffer, ns * ms * sizeof(float));
        }
        else
        {
            buffer = new uint8_t[n * m * sizeof(float)];
            float *floatBuffer = (float *)buffer;

            FILE *fi = fopen(this->fileName.c_str(), "rb");
            int ret;
#if defined(_WIN32) || defined(_WIN64)
            _fseeki64(fi, this->data_offsets[0], 0);
#else
            fseek(fi, this->data_offsets[0], 0);
#endif
            uint8_t *ori = new uint8_t[this->bytes];
            ret = fread(ori, 1, this->bytes, fi);
            for (int bi = 0; bi < ns; bi++)
            {
                for (int bj = 0; bj < ms; bj++)
                {
                    float curScale = ((float *)scale.buffer)[bi * ms + bj];
                    for (int i = bi * blockN; i < (bi + 1) * blockN && i < n; i++)
                    {
                        for (int j = bj * blockM; j < (bj + 1) * blockM && j < m; j++)
                        {
                            floatBuffer[i * m + j] = curScale * fp8e4m3tofp32.dict[ori[i * m + j]];
                        }
                    }
                }
            }

            delete[] ori;
            fclose(fi);
        }
    }

    void CreateBufferWithAWQ(DataType dstType, SafeTensorItem &scale, SafeTensorItem &qzero)
    {
        const int groupCnt = this->shape[0] / scale.shape[0];
        AssertInFastLLM(this->shape.size() == 2 && scale.shape.size() == 2 && qzero.shape.size() == 2,
                        "CreateBufferWithAWQ error: shape.size() should be 2.");
        AssertInFastLLM(groupCnt * scale.shape[0] == this->shape[0] && groupCnt * qzero.shape[0] == this->shape[0] &&
                            8 * this->shape[1] == scale.shape[1] && this->shape[1] == qzero.shape[1],
                        "CreateBufferWithAWQ error: shape error.");
        AssertInFastLLM(this->dtype == "I32" && qzero.dtype == "I32",
                        "CreateBufferWithAWQ error: dtype shoud be I32.");
        int n = this->shape[0], m = this->shape[1];

        ClearBuffer();
        FILE *fweight = fopen(this->fileName.c_str(), "rb");
        FILE *fqzero = fopen(qzero.fileName.c_str(), "rb");
#if defined(_WIN32) || defined(_WIN64)
        _fseeki64(fweight, this->data_offsets[0], 0);
        _fseeki64(fqzero, qzero.data_offsets[0], 0);
#else
        fseek(fweight, this->data_offsets[0], 0);
        fseek(fqzero, qzero.data_offsets[0], 0);
#endif
        uint8_t *ori_weight = new uint8_t[this->bytes];
        uint8_t *ori_qzero = new uint8_t[qzero.bytes];
        int ret;
        ret = fread(ori_weight, 1, this->bytes, fweight);
        ret = fread(ori_qzero, 1, qzero.bytes, fqzero);
        unsigned int *weight_int32 = (unsigned int *)ori_weight;
        unsigned int *qzero_int32 = (unsigned int *)ori_qzero;
        float *scale_f32 = (float *)scale.buffer;
        static const int awq_shift[8] = {0, 16, 4, 20, 8, 24, 12, 28}; // awq order = [0,2,4,8,1,3,5,7]

        if (dstType == DataType::FLOAT32)
        {
            buffer = new uint8_t[this->bytes * 8];
            float *floatBuffer = (float *)buffer;
            for (int x = 0; x < n; x++)
            {
                for (int y = 0; y < m * 8; y++)
                {
                    int gx = x / groupCnt;
                    int gy = y >> 3;
                    int w = (weight_int32[x * m + gy] >> awq_shift[y & 7]) & 15;
                    int z = (qzero_int32[gx * m + gy] >> awq_shift[y & 7]) & 15;
                    float s = scale_f32[gx * m * 8 + y];
                    floatBuffer[y * n + x] = (w - z) * s;
                }
            }
        }
        else if (dstType == DataType::INT4_GROUP)
        {
            buffer = new uint8_t[this->bytes];
            memset(buffer, 0, this->bytes);
            int group = (n - 1) / groupCnt + 1;
            scalesBuffer = new float[m * 8 * group];
            minsBuffer = new float[m * 8 * group];
            for (int x = 0; x < n; x += groupCnt)
            {
                for (int y = 0; y < m * 8; y++)
                {
                    int gx = x / groupCnt;
                    int gy = y >> 3;
                    int z = (qzero_int32[gx * m + gy] >> awq_shift[y & 7]) & 15;
                    float s = scale_f32[gx * m * 8 + y];
                    scalesBuffer[y * group + x / groupCnt] = s;
                    minsBuffer[y * group + x / groupCnt] = -s * z;
                }
            }
            for (int x = 0; x < n; x++)
            {
                for (int y = 0; y < m * 8; y++)
                {
                    int gx = x / groupCnt;
                    int gy = y >> 3;
                    int w = (weight_int32[x * m + gy] >> awq_shift[y & 7]) & 15;
                    buffer[y * n / 2 + x / 2] += (w << ((1 - (x & 1)) * 4));
                }
            }
        }
        else
        {
            ErrorInFastLLM_A("CreateBufferWithAWQ Error: dst type error.");
        }
        delete[] ori_weight;
        delete[] ori_qzero;
        fclose(fweight);
        fclose(fqzero);
    }

    void CreateBuffer(DataType dstType)
    {
        // printf("read %s from %s [%llu %llu] (%f M)\n", this->tensorName.c_str(), this->fileName.c_str(), this->data_offsets[0], this->data_offsets[0] + this->bytes, (float)this->bytes / 1e6);
        FILE *fi = fopen(this->fileName.c_str(), "rb");
        int ret;
#if defined(_WIN32) || defined(_WIN64)
        _fseeki64(fi, this->data_offsets[0], 0);
#else
        fseek(fi, this->data_offsets[0], 0);
#endif
        DataType srcType;
        if (this->dtype == "fastllm")
        {
            ClearBuffer();
            buffer = new uint8_t[this->bytes];
            ret = fread(buffer, 1, this->bytes, fi);
            fclose(fi);
            return;
        }
        else if (this->dtype == "F8_E4M3")
        {
            srcType = DataType::FP8_E4M3;
        }
        else if (this->dtype == "BF16")
        {
            srcType = DataType::BFLOAT16;
        }
        else if (this->dtype == "F16")
        {
            srcType = DataType::FLOAT16;
        }
        else if (this->dtype == "F32")
        {
            srcType = DataType::FLOAT32;
            if (dstType != DataType::FLOAT32)
            {
                ErrorInFastLLM_A("SafeTensorItem.CreateBuffer: unsupport src dtype " + this->dtype + "\n");
            }
        }
        else if (this->dtype == "I64")
        {
            printf("skip I64 tensor %s\n", this->tensorName.c_str());
            return;
        }
        else
        {
            ErrorInFastLLM_A("SafeTensorItem.CreateBuffer: unsupport src dtype " + this->dtype + "\n");
        }

        int unitSize = 4;
        if (dstType == DataType::FLOAT32)
        {
            unitSize = 4;
        }
        else if (dstType == DataType::FLOAT16 || dstType == DataType::BFLOAT16)
        {
            unitSize = 2;
        }
        else
        {
            ErrorInFastLLM_A("SafeTensorItem.CreateBuffer: unsupport dst dtype " + std::to_string(dstType) + "\n");
        }
        ClearBuffer();
        buffer = new uint8_t[(size_t)len * unitSize];
        if (dstType == srcType)
        {
            ret = fread(buffer, 1, this->bytes, fi);
        }
        else
        {
            uint8_t *ori = new uint8_t[this->bytes];
            ret = fread(ori, 1, this->bytes, fi);
            ConvertDataType(ori, srcType, buffer, dstType, len);
            delete[] ori;
        }
        fclose(fi);
    }

    void Transpose(DataType type)
    {
        int n = intShape[0], m = intShape[1];
        if (type == DataType::FLOAT32)
        {
            float *temp = new float[len];
            memcpy(temp, this->buffer, len * sizeof(float));
            Transpose_A((float *)this->buffer, temp, n, m, n, m);
            delete[] temp;
        }
        else if (type == DataType::FLOAT16 || type == DataType::BFLOAT16)
        {
            uint16_t *temp = new uint16_t[len];
            memcpy(temp, this->buffer, len * sizeof(uint16_t));
            TransposeSimple((uint16_t *)this->buffer, temp, n, m, n, m);
            delete[] temp;
        }
        else
        {
            ErrorInFastLLM_A("SafeTensorItem.Transpose: unsupport dtype " + std::to_string(type) + "\n");
        }
    }

    void ClearBuffer()
    {
        delete[] buffer;
        buffer = nullptr;
        delete[] minsBuffer;
        minsBuffer = nullptr;
        delete[] scalesBuffer;
        scalesBuffer = nullptr;
    }
};

struct SafeTensors
{
    std::set<std::string> fileNames;
    std::map<std::string, SafeTensorItem> itmeDict;

    SafeTensors(const std::set<std::string> &fileNames)
    {
        std::string error;
        this->fileNames = fileNames;
        for (auto &fileName : fileNames)
        {
            FILE *f = fopen(fileName.c_str(), "rb");
            uint64_t configBytes;
            int ret = fread(&configBytes, 8, 1, f);
            std::cout << "configBytes: " << configBytes << std::endl;
            char *configString = new char[configBytes + 5];
            ret = fread(configString, 1, configBytes, f);
            configString[configBytes] = 0;
            
            auto config = json11::Json::parse(configString, error);
            std::cout << "config: " << config.dump() << std::endl;
            for (auto it : config.object_items())
            {
                if (it.first != "__metadata__")
                {
                    itmeDict[it.first] = SafeTensorItem(it.first, fileName, it.second, 8 + configBytes);
                }
            }

            delete[] configString;
        }
    }

    std::vector<std::string> GetSortedItemNames()
    {
        std::vector<std::pair<std::pair<std::string, uint64_t>, std::string>> v;
        for (auto &it : itmeDict)
        {
            if (it.second.intShape.size() > 0 && it.second.dtype != "BOOL")
            {
                v.push_back(std::make_pair(std::make_pair(it.second.fileName, it.second.data_offsets[0]), it.first));
            }
        }
        std::sort(v.begin(), v.end());
        std::vector<std::string> ret;
        for (int i = 0; i < v.size(); i++)
        {
            ret.push_back(v[i].second);
        }
        return ret;
    }
};

int main()
{
    std::map<std::string, std::pair<std::string, std::string>> loraDicts;
    SafeTensors *loraTensors = nullptr;
    loraTensors = new SafeTensors({"/data1/temp/llm_lora/snshrivas10/sft-tiny-chatbot/adapter_model.safetensors"});

    for (auto &it : loraTensors->GetSortedItemNames())
    {
        if (it.size() >= 31 &&
            it.substr(0, 17) == "base_model.model." &&
            (it.substr(it.size() - 14) == ".lora_A.weight" || it.substr(it.size() - 14) == ".lora_B.weight"))
        {
            std::string originalName = it.substr(17, it.size() - 31) + ".weight";
            if (it.substr(it.size() - 14) == ".lora_A.weight")
            {
                loraDicts[originalName].first = it;
            }
            else
            {
                loraDicts[originalName].second = it;
            }
        }
    }

    return 0; // 成功返回 0
}