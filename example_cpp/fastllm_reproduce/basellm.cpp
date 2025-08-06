#include "basellm.h"
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "file_utils.hpp"
#include "qwen3.h"
#include <cstring>

basellm::basellm() {}

basellm::~basellm() {}

void basellm::InitParams() {
    if (this->weight.dicts.find("model_type") != this->weight.dicts.end()) {
        this->model_type = this->weight.dicts["model_type"].c_str();
    }

    if (this->weight.dicts.find("bos_token_id") != this->weight.dicts.end()) {
        this->bos_token_id = atoi(this->weight.dicts["bos_token_id"].c_str());
    }

    if (this->weight.dicts.find("eos_token_id") != this->weight.dicts.end()) {
        this->eos_token_id = atoi(this->weight.dicts["eos_token_id"].c_str());
    }

    if (this->weight.dicts.find("hidden_size") != this->weight.dicts.end()) {
        this->embed_dim = atoi(this->weight.dicts["hidden_size"].c_str());
    }

    if (this->weight.dicts.find("num_hidden_layers") != this->weight.dicts.end()) {
        this->block_cnt = atoi(this->weight.dicts["num_hidden_layers"].c_str());
    }

    if (this->weight.dicts.find("head_dim") != this->weight.dicts.end()) {
        this->head_dim = atoi(this->weight.dicts["head_dim"].c_str());
        this->rotary_dim = this->head_dim;
    }

    if (this->weight.dicts.find("num_attention_heads") != this->weight.dicts.end()) {
        this->num_attention_heads = atoi(this->weight.dicts["num_attention_heads"].c_str());
    }

    this->embed_dim = this->head_dim * this->num_attention_heads;

    this->num_key_value_heads = this->num_attention_heads;
    if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
        this->num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
    }
}

std::map<std::string, std::vector<std::pair<std::string, DataType>>> basellm::GetTensorMap(const std::vector<std::string> &tensorNames) {
    std::map<std::string, std::vector<std::pair<std::string, DataType>>> ret;
    for (auto &name : tensorNames) {
        std::string realName = name;
        if (StringEndWith(name, "qweight")) {
            realName = name.substr(0, name.size() - 7) + "weight";
        }

        WeightType weightType = this->weight.GetWeightType(realName);

        DataType dataType = DataType::DATA_AUTO_NONE;
        if (weightType == WeightType::LINEAR) {
            dataType = DataType::DATA_AUTO_LINEAR;
            if (this->cantQuantLinears.find(realName) != this->cantQuantLinears.end()) {
                dataType = DataType::BFLOAT16;
            }
        } else if (weightType == WeightType::EMBEDDING) {
            dataType = DataType::DATA_AUTO_EMBEDDING;
        }
        ret[name].push_back(std::make_pair(realName, dataType));
    }

    return ret;
}

std::map<std::string, std::vector<std::pair<std::string, DataType>>>
basellm::GetTensorMap(const std::vector<std::string> &tensorNames, bool useMoeDataType, DataType moeDataType) {
    std::map<std::string, std::vector<std::pair<std::string, DataType>>> ret;

    for (auto &name : tensorNames) {
        std::string realName = name;
        if (StringEndWith(name, ".qweight")) {
            realName = name.substr(0, name.size() - 7) + "weight";
        }

        WeightType weightType = this->weight.GetWeightType(realName);
        DataType dataType = DataType::DATA_AUTO_NONE;

        if (weightType == WeightType::LINEAR) {
            dataType = DataType::DATA_AUTO_LINEAR;
            if (this->cantQuantLinears.find(realName) != this->cantQuantLinears.end()) {
                dataType = DataType::FLOAT16;
            }
        } else if (weightType == WeightType::EMBEDDING) {
            dataType = DataType::DATA_AUTO_EMBEDDING;
        }

        // 如果是 MoE 并且开启 useMoeDataType，则替换类型
        if (useMoeDataType && this->moeLinears.find(realName) != this->moeLinears.end()) {
            dataType = moeDataType;
        }

        ret[name].push_back(std::make_pair(realName, dataType));
    }

    return ret;
}

Data::Data() {}

Data::Data(DataType datatype) {
    this->dataType = datatype;
    this->UpdateUnitSize();
}

Data::Data(DataType datatype, const std::vector<int> &dims) {
    this->dataType = datatype;
    this->Resize(dims);
}

Data::Data(DataType datatype, const std::vector<int> &dims, DataDevice device, void *ptr) : Data::Data(datatype, dims) {
    this->isFake = true;
    this->expansionSize = this->Count(0);
    this->UpdateUnitSize();
    this->dataDevice = device;
    if (this->dataDevice == DataDevice::CPU) {
        this->cpuData = (uint8_t *)ptr;
    } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
        this->cudadata = ptr;
        this->dataDeviceIds = {0};
#endif
        ErrorInFastLLM("Error: cuda is not supported.\n");
    }
}

Data::Data(DataType datatype, const std::vector<int> &dims, const std::vector<float> data) : Data::Data(datatype, dims) {
    this->Allocate();
    if (datatype == DataType::FLOAT32) {
        std::memcpy(this->cpuData, data.data(), this->GetBytes());
    }
}

void Data::UpdateUnitSize() {
    if (this->dataType == DataType::FLOAT32 || this->dataType == DataType::INT32PARAM) {
        this->unitSize = 4;
        this->unitSizeDiv = 1;
    } else if (this->dataType == DataType::BFLOAT16 || this->dataType == DataType::INT16 || this->dataType == DataType::FLOAT16) {
        this->unitSize = 2;
        this->unitSizeDiv = 1;
    } else if (this->dataType == DataType::INT8 || this->dataType == DataType::FP8_E4M3) {
        this->unitSize = 1;
        this->unitSizeDiv = 1;
    } else if (this->dataType == DataType::INT4 || this->dataType == DataType::INT4_NOZERO || this->dataType == DataType::INT4_GROUP) {
        this->unitSize = 1;
        this->unitSizeDiv = 2;
    } else if (this->dataType == DataType::INT2 || this->dataType == DataType::INT2_GROUP) {
        this->unitSize = 1;
        this->unitSizeDiv = 4;
    } else if (this->dataType == DataType::BIT) {
        this->unitSize = 1;
        this->unitSizeDiv = 8;
    }

    this->expansionBytes = (this->expansionSize * this->unitSize - 1) / this->unitSizeDiv + 1;
}

void Data::Resize(const std::vector<int> &dims) {
    this->dims = dims;
    this->UpdateUnitSize();

    if (this->expansionDims.size() == 0) {
        this->stride.resize(this->dims.size(), 1);
        this->stride.back() = 1;
        for (int i = this->stride.size() - 2; i >= 0; i--) {
            this->stride[i] = this->stride[i + 1] * this->dims[i + 1];
        }
    }
}

uint64_t Data::Count(int i) const {
    if (i >= this->stride.size()) {
        return 1;
    }

    if ((i - 1 >= 0) && (i - 1 < this->stride.size())) {
        return this->stride[i - 1];
    }

    return this->dims[i] * this->stride[i];
}

uint64_t Data::GetBytes() const { return (this->dims[0] * this->stride[0] * this->unitSize - 1) / this->unitSizeDiv - 1; }

void Data::FreeSpace() {
    this->expansionSize = 0;
    this->expansionBytes = 0;
    if (this->dataDevice == DataDevice::CPU) {
        delete[] this->cpuData;
    } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
        if (this->directMemory) {
            FastllmCudaDirectFree(this->cudaData);
        } else {
            FastllmCudaFree(this->cudaData);
        }
#else
        ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
    }
}

void Data::Allocate() {
    if (!this->isFake && this->Count(0) > this->expansionSize) {
        this->FreeSpace();
        this->MallocSpace(this->Count(0));
    }
}

void Data::MallocSpace(uint64_t size_t) {
    this->expansionSize = size_t;
    this->expansionBytes = (this->expansionSize * this->unitSize - 1) / this->unitSizeDiv + 1;
    if (this->dataDevice == DataDevice::CPU) {
        this->cpuData = new uint8_t[this->expansionBytes];
        std::memset(this->cpuData, 0, this->expansionBytes * sizeof(uint8_t));
    }
}

void Data::Expansion(const std::vector<int> &dims) {
    if (this->dims.size() == 0) {
        this->directMemory = true;
        this->expansionDims = dims;
        this->stride.resize(dims.size(), 1);
        this->stride.back() = 1;
        for (int i = this->stride.size() - 2; i >= 0; i--) {
            this->stride[i] = this->stride[i + 1] * this->dims[i + 1];
        }
        this->MallocSpace(dims[0] * this->stride[0]);
        return;
    }

    AssertInFastLLM(dims.size() == this->dims.size(), "Expansion error: real dims's size should equal to expansion dims's size.\n");
    for (int i = 0; i < dims.size(); i++) {
        AssertInFastLLM(dims[i] == -1 || dims[i] >= this->dims[i], "Expansion error: real size should <= expansion size.\n");
    }

    int axis = -1;
    for (int i = 0; i < this->dims.size(); i++) {
        if (this->dims[i] < dims[i]) {
            axis = i;
            break;
        }
    }

    int input0stride = this->Count(axis);
    this->expansionDims = dims;

    this->stride.resize(dims.size(), 1);
    this->stride.back() = 1;
    for (int i = this->stride.size() - 2; i >= 0; i--) {
        this->stride[i] = this->stride[i + 1] * dims[i + 1];
    }

    if (this->expansionBytes != 0) {
        if (this->dataDevice == DataDevice::CPU) {
            uint8_t *old = this->cpuData;
            this->MallocSpace(dims[0] * this->stride[0]);
            int outer = this->Count(0) / this->Count(axis);
            int input1stride = this->Count(axis);
            int unitSize = this->unitSize;
            int inner = this->stride[axis];
            for (int o = 0; o < outer; o++) {
                std::memcpy(this->cpuData + o * input1stride * unitSize, old + o * input0stride * unitSize, this->dims[axis] * inner * unitSize);
            }
        }
    } else {
        this->MallocSpace(dims[0] * this->stride[0]);
    }
}

void Data::ToDevice(DataDevice device) {
    if (device == DataDevice::CUDA) {

    } else if (device == DataDevice::CPU) {
        std::vector<int> deviceIds = {0};
        this->ToDevice(device, deviceIds);
    }
}

void Data::ToDevice(DataDevice device, std::vector<int> &deviceIds) {
    if (this->dataType == DataType::INT32PARAM) {
        return;
    }
#ifndef USE_CUDA
    return;
#endif

    if (this->dataDevice == device && (device == DataDevice::CPU || deviceIds.size() == 0 || this->dataDeviceIds == deviceIds)) {
        return;
    }

    if (this->expansionBytes != 0) {

#ifdef USE_CUDA
        if (this->dataDevice == DataDevice::CPU && device == DataDevice::CUDA) {
            uint8_t *cpuData = this->cpuData;
#ifdef USE_MMAP
            cpuData = new uint8_t[expansionBytes];
            memcpy(cpuData, this->cpuData, expansionBytes);
#endif
            // FastllmCudaSetDevice(deviceIds.size() == 0 ? 0 : deviceIds[0]);
            this->cudaData = FastllmCudaMalloc(expansionBytes);
            FastllmCudaCopyFromHostToDevice(this->cudaData, cpuData, expansionBytes);
#ifdef USE_MMAP
            delete[] cpuData;
#else
            delete[] this->cpuData;
            this->cpuData = nullptr;
#endif
        } else if (this->dataDevice == DataDevice::CUDA) {
            if (device == DataDevice::CPU) {
                this->cpuData = new uint8_t[this->expansionBytes];
                FastllmCudaCopyFromDeviceToHost(this->cpuData, this->cudaData, this->expansionBytes);
                FastllmCudaFree(this->cudaData);
                this->cudaData = nullptr;
            } else if (device == DataDevice::CUDA) {
                int sourceDevice = this->dataDeviceIds.size() == 0 ? 0 : this->dataDeviceIds[0];
                int destDevice = deviceIds.size() == 0 ? 0 : deviceIds[0];
                if (sourceDevice != destDevice) {
                    FastllmCudaSetDevice(destDevice);
                    void *newCudaData = FastllmCudaMalloc(this->expansionBytes);
                    FastllmCudaMemcpyBetweenDevices(destDevice, newCudaData, sourceDevice, this->cudaData, this->expansionBytes);
                    FastllmCudaSetDevice(sourceDevice);
                    FastllmCudaFree(this->cudaData);
                    this->cudaData = newCudaData;
                }
            }
        }
#endif
    }

    this->dataDevice = device;
    if (deviceIds.size() == 0) {
        this->dataDeviceIds = {0};
    } else {
        this->dataDeviceIds = deviceIds;
    }
}

void Data::CopyFrom(const Data &ori) {

    this->ToDevice(ori.dataDevice);
    this->name = ori.name;
    this->isKVCache = ori.isKVCache;
    this->cacheUid = ori.cacheUid;
    this->dataDevice = ori.dataDevice;

    if (ori.expansionDims != this->expansionDims || ori.dims != this->dims || this->cpuData == nullptr || ori.dataType != this->dataType) {
        if (ori.dims.size() == 0) {
            this->dataType = ori.dataType;
            this->UpdateUnitSize();
            this->dims.resize(0);

            if (this->dataDevice == DataDevice::CPU) {
                delete[] this->cpuData;
                this->cpuData = nullptr;
            } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
                FastllmCudaFree(this->cudaData);
                this->cudaData = nullptr;
#endif
            }
            return;
        }

        this->dataType = ori.dataType;
        this->UpdateUnitSize();
        if (ori.expansionDims.size() > 0 && ori.expansionDims != ori.dims) {
            this->Expansion(ori.expansionDims);
            this->Resize(ori.dims);
            this->Allocate();
        } else {
            this->expansionDims.clear();
            this->Resize(ori.dims);
            this->FreeSpace();
            this->MallocSpace(Count(0));
        }
    }

    if (this->dataDevice == DataDevice::CPU) {
        std::memcpy(this->cpuData, ori.cpuData, this->GetBytes());
    } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
        FastllmCudaCopyFromDeviceToDevice(this->cudaData, ori.cudaData, this->GetBytes());
#endif
    }
}

BF16ToFP16Manager bf16tofp16;

void Data::CreateFromOriData(
    WeightType weightType, DataType oriDataType, uint8_t *oriData, float *oriMins, float *oriScales, int groupCnt, int blockK, int blockM) {
    this->weightType = weightType;
    this->UpdateUnitSize();
    this->Allocate();
    if (this->dataType == oriDataType) {
        if (oriData != nullptr) {
            memcpy(this->cpuData, oriData, this->GetBytes());
        }
        if (oriDataType == DataType::INT4_GROUP) {
            int k = this->dims[0], m = this->dims[1], group = (m - 1) / groupCnt + 1;
            this->group = group;
            this->groupCnt = groupCnt;
            this->scales.resize(k * group);
            this->mins.resize(k * group);

            memcpy(this->scales.data(), oriScales, k * group * sizeof(float));
            memcpy(this->mins.data(), oriMins, k * group * sizeof(float));

            this->perChannelAxis = 0;
        } else if (oriDataType == DataType::FP8_E4M3) {
            int k = this->dims[0], m = this->dims[1];

            this->blockK = blockK;
            this->blockM = blockM;

            int ks = (k - 1) / this->blockK + 1;
            int ms = (m - 1) / this->blockM + 1;

            this->scales.resize(ks * ms);
            memcpy(this->scales.data(), oriScales, ks * ms * sizeof(float));
        }
    } else if (oriDataType == DataType::BFLOAT16 && this->dataType == DataType::FLOAT16) {
        uint16_t *b = (uint16_t *)oriData;
        uint16_t *a = (uint16_t *)this->cpuData;
        int len = this->Count(0);
        for (int i = 0; i < len; i++) {
            a[i] = bf16tofp16.dict[b[i]];
        }
    } else if (oriDataType == DataType::FLOAT32 && this->dataType == DataType::FLOAT16) {
        float *b = (float *)oriData;
        uint16_t *a = (uint16_t *)this->cpuData;
        int len = this->Count(0);
        for (int i = 0; i < len; i++) {
            a[i] = float_to_half(b[i]);
        }
    } else if ((oriDataType == DataType::FLOAT32 || oriDataType == DataType::BFLOAT16) && this->dataType == DataType::INT4_GROUP) {
        int bit = 4;
        int type = 1;
        int k = this->dims[0], m = this->dims[1], group = (m - 1) / groupCnt + 1;
        if (groupCnt == -1) {
            groupCnt = 128;
        }
        std::vector<LowBitConfig> configs;
        std::vector<uint8_t> uDatas;
        this->group = group;
        this->groupCnt = groupCnt;
        this->perChannelAxis = 0;
        this->scales.resize(k * group);
        this->mins.resize(k * group);

        int bytes = (k * m + 1) / 2;

        configs.resize(k * group);
        uDatas.resize(bytes);

        if (oriDataType == DataType::FLOAT32) {
            MultiThreadGroupQuantizationOp(0, k, m, bit, configs.data(), group, groupCnt, (float *)oriData, uDatas.data(), type).Run();
        } else {
            MultiThreadGroupQuantizationBF16Op(0, k, m, (uint16_t *)oriData, uDatas.data(), configs.data(), bit, group, groupCnt, type).Run();
        }

        for (int i = 0; i < k * group; i++) {
            float min = configs[i].min;
            float scale = configs[i].scale;

            this->mins[i] = min;
            this->scales[i] = scale;
        }

        memcpy(this->cpuData, uDatas.data(), bytes);
    } else if ((oriDataType == DataType::FLOAT32 || oriDataType == DataType::BFLOAT16) && this->dataType == DataType::INT2_GROUP) {
        int bit = 4;
        int type = 1;
        int k = this->dims[0], m = this->dims[1], group = (m - 1) / groupCnt + 1;
        if (groupCnt == -1) {
            groupCnt = 32;
        }
        std::vector<LowBitConfig> configs;
        std::vector<uint8_t> uDatas;
        this->group = group;
        this->groupCnt = groupCnt;
        this->perChannelAxis = 0;
        this->scales.resize(k * group);
        this->mins.resize(k * group);

        int bytes = k * m / 4;
        if (k * m % 4 != 0) {
            bytes++;
        }

        configs.resize(k * group);
        uDatas.resize(bytes);

        if (oriDataType == DataType::FLOAT32) {
            MultiThreadGroupQuantizationOp(0, k, m, bit, configs.data(), group, groupCnt, (float *)oriData, uDatas.data(), type).Run();
        } else {
            MultiThreadGroupQuantizationBF16Op(0, k, m, (uint16_t *)oriData, uDatas.data(), configs.data(), bit, group, groupCnt, type).Run();
        }

        for (int i = 0; i < k * group; i++) {
            float min = configs[i].min;
            float scale = configs[i].scale;

            this->mins[i] = min;
            this->scales[i] = scale;
        }

        memcpy(this->cpuData, uDatas.data(), bytes);
    } else if ((oriDataType == DataType::FLOAT32 || oriDataType == DataType::BFLOAT16) &&
               (this->dataType == DataType::INT8 || this->dataType == DataType::INT4_NOZERO)) {
        int bit = (this->dataType == INT4_NOZERO) ? 4 : 8;
        int type = (bit == 4) ? 1 : 0;
        int k = this->dims[0], m = this->dims[1];

        int bytes = (k * m + 1) / 2;
        if (bit == 8) {
            bytes = k * m;
        }

        std::vector<uint8_t> uDatas;
        std::vector<LowBitConfig> configs;

        this->scales.resize(k);
        this->mins.resize(k);
        this->zeros.resize(k);
        this->perChannelsConfigs.resize(k);
        uDatas.resize(bytes);
        configs.resize(k);

        this->perChannelAxis = 0;

        if (oriDataType == DataType::FLOAT32) {
            MultiThreadPerChannelQuantizationOp(0, k, m, (float *)oriData, uDatas.data(), configs.data(), bit, type).Run();
        } else {
            MultiThreadPerChannelQuantizationBF16Op(0, k, m, (uint16_t *)oriData, uDatas.data(), configs.data(), bit, type).Run();
        }

        for (int i = 0; i < k; i++) {
            this->perChannelsConfigs[i] = configs[i];
            this->scales[i] = configs[i].scale;
            this->mins[i] = configs[i].min;
            this->zeros[i] = configs[i].zeroPoint;
        }

        memcpy(this->cpuData, uDatas.data(), bytes);
    } else if ((oriDataType == DataType::FLOAT32 || oriDataType == DataType::BFLOAT16) && this->dataType == DataType::BASE3_GROUP) {
        int k = this->dims[0], m = this->dims[1];
        if (groupCnt == -1) {
            groupCnt = 128;
        }
        int group = (m - 1) / groupCnt + 1;
        int bytesPerGroup = (groupCnt - 1) / 5 + 1;

        this->group = group;
        this->groupCnt = groupCnt;

        std::vector<uint8_t> uDatas;

        uDatas.resize(k * group * bytesPerGroup);
        this->halfScales.resize(k * group);

        if (oriDataType == DataType::FLOAT32) {
            MultiThreadBase3GroupQuantizationOp(0, k, m, (float *)oriData, uDatas.data(), this->halfScales.data(), group, groupCnt).Run();
        } else {
            MultiThreadBase3GroupQuantizationBF16Op(0, k, m, (uint16_t *)oriData, uDatas.data(), this->halfScales.data(), group, groupCnt).Run();
        }

        memcpy(this->cpuData, uDatas.data(), k * group * bytesPerGroup);
    } else {
        ErrorInFastLLM("wrong data type " + dataTypeNames[oriDataType][0] + " -> " + dataTypeNames[dataType][0]);
    }
}

void Data::ExportFastllmFormat(uint8_t *bytes) {
    ByteWriter writer(bytes);
    if (this->dataType == DataType::BFLOAT16 || this->dataType == DataType::FLOAT16 || this->dataType == DataType::FLOAT32) {
        writer.WriteBytes(this->cpuData, this->GetBytes());
        return;
    }

    writer.WriteInt(1);
    writer.WriteInt((int)this->dataType);
    if (this->dataType == DataType::FP8_E4M3) {
        writer.WriteInt(this->blockK);
        writer.WriteInt(this->blockM);
        writer.WriteInt(this->scales.size());
        writer.WriteBytes((uint8_t *)this->scales.data(), this->scales.size() * sizeof(float));
        writer.WriteBytes(this->cpuData, this->GetBytes());
    } else if (this->dataType == DataType::INT8 || this->dataType == DataType::INT4 || this->dataType == DataType::INT4_NOZERO) {
        writer.WriteInt(this->perChannelAxis);
        int k = this->perChannelAxis == -1 ? 1 : this->dims[perChannelAxis];
        for (int i = 0; i < k; i++) {
            writer.WriteFloat(this->perChannelsConfigs[i].min);
            if (this->dataType == DataType::INT4_NOZERO) {
                writer.WriteFloat(this->perChannelsConfigs[i].scale);
            } else {
                writer.WriteFloat(this->perChannelsConfigs[i].max);
            }
        }

        writer.WriteBytes(this->cpuData, this->GetBytes());
    } else if (this->dataType == DataType::INT4_GROUP) {
        writer.WriteInt(this->perChannelAxis);
        writer.WriteInt(this->group);
        writer.WriteInt(this->groupCnt);
        int k = this->perChannelAxis == -1 ? 1 : this->dims[perChannelAxis];
        for (int i = 0; i < k * this->group; i++) {
            writer.WriteFloat(this->mins[i]);
            writer.WriteFloat(this->scales[i]);
        }
        writer.WriteBytes(this->cpuData, this->GetBytes());
    } else {
        ErrorInFastLLM("ExportFastllmFormat Error: data type error.");
    }
}

uint64_t Data::GetFastllmFormateBytes() {
    if (this->dataType == DataType::BFLOAT16 || this->dataType == DataType::FLOAT16 || this->dataType == DataType::FLOAT32) {
        return this->GetBytes();
    }

    int ret = 0;
    ret += sizeof(int) * 2;
    if (this->dataType == DataType::FP8_E4M3) {
        ret += sizeof(int) * 3;
        ret += this->scales.size() * sizeof(float);
        ret += this->GetBytes();
    } else if (this->dataType == DataType::INT8 || this->dataType == DataType::INT4 || this->dataType == DataType::INT4_NOZERO) {
        ret += sizeof(int);
        int k = this->perChannelAxis == -1 ? 1 : this->dims[perChannelAxis];
        ret += k * 2 * sizeof(float);
        ret += this->GetBytes();
    } else if (this->dataType == DataType::INT4_GROUP) {
        ret += sizeof(int) * 3;
        int k = this->perChannelAxis == -1 ? 1 : this->dims[perChannelAxis];
        ret += k * this->group * 2 * sizeof(float);
        ret += this->GetBytes();
    } else {
        ErrorInFastLLM("ExportFastllmFormat Error: data type error.");
    }

    return ret;
}