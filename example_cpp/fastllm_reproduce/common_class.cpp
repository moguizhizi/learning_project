#include "common_class.h"

#include <cstring>

#include "computeutils.h"
#include "cpudevice.h"
#include "fastllm.h"
#include "file_utils.hpp"
#include "struct_space.hpp"
#include "utils.h"

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
        this->strides.resize(this->dims.size(), 1);
        this->strides.back() = 1;
        for (int i = this->strides.size() - 2; i >= 0; i--) {
            this->strides[i] = this->strides[i + 1] * this->dims[i + 1];
        }
    }
}

uint64_t Data::Count(int i) const {
    if (i >= this->strides.size()) {
        return 1;
    }

    if ((i - 1 >= 0) && (i - 1 < this->strides.size())) {
        return this->strides[i - 1];
    }

    return this->dims[i] * this->strides[i];
}

uint64_t Data::GetBytes() const {
    return (this->dims[0] * this->strides[0] * this->unitSize - 1) / this->unitSizeDiv - 1;
}

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

void Data::Allocate(float v) {
    AssertInFastLLM(this->dataType == DataType::FLOAT32 || this->dataType == DataType::FLOAT16,
        "Allocate error: Data's type should be float32 or float16.\n");
    this->Allocate();
    if (this->dataDevice == DataDevice::CPU) {
        if (this->dataType == DataType::FLOAT32) {
            float *f = (float *)cpuData;
            std::fill(f, f + Count(0), v);
        } else if (this->dataType == DataType::FLOAT16) {
            uint16_t *h = (uint16_t *)cpuData;
            std::fill(h, h + Count(0), float_to_half(v));
        }
    }
    if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
        if (this->dataType == DataType::FLOAT32) {
            std::vector<float> f = std::vector<float>(Count(0), v);
            FastllmCudaCopyFromHostToDevice(cudaData, f.data(), Count(0) * sizeof(float));
        } else if (this->dataType == DataType::FLOAT16) {
            std::vector<uint16_t> f = std::vector<uint16_t>(Count(0), float_to_half(v));
            FastllmCudaCopyFromHostToDevice(cudaData, f.data(), Count(0) * sizeof(uint16_t));
        }
#endif
    } else {
        // TODO: 别的设备上的初始化
    }
}

void Data::MallocSpace(uint64_t size_t) {
    this->expansionSize = size_t;
    this->expansionBytes = (this->expansionSize * this->unitSize - 1) / this->unitSizeDiv + 1;
    if (this->dataDevice == DataDevice::CPU) {
        this->cpuData = new uint8_t[this->expansionBytes];
        std::memset(this->cpuData, 0, this->expansionBytes * sizeof(uint8_t));
    } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
        if (this->directMemory) {
            this->cudaData = FastllmCudaDirectMalloc(this->expansionBytes);
        } else {
            this->cudaData = FastllmCudaMalloc(this->expansionBytes);
        }
        FastllmCudaMemset0(this->cudaData, this->expansionBytes);
#else
        ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
    }
}

void Data::Expansion(const std::vector<int> &dims) {
    if (this->dims.size() == 0) {
        this->directMemory = true;
        this->expansionDims = dims;
        this->strides.resize(dims.size(), 1);
        this->strides.back() = 1;
        for (int i = this->strides.size() - 2; i >= 0; i--) {
            this->strides[i] = this->strides[i + 1] * this->dims[i + 1];
        }
        this->MallocSpace(dims[0] * this->strides[0]);
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

    this->strides.resize(dims.size(), 1);
    this->strides.back() = 1;
    for (int i = this->strides.size() - 2; i >= 0; i--) {
        this->strides[i] = this->strides[i + 1] * dims[i + 1];
    }

    if (this->expansionBytes != 0) {
        if (this->dataDevice == DataDevice::CPU) {
            uint8_t *old = this->cpuData;
            this->MallocSpace(dims[0] * this->strides[0]);
            int outer = this->Count(0) / this->Count(axis);
            int input1stride = this->Count(axis);
            int unitSize = this->unitSize;
            int inner = this->strides[axis];
            for (int o = 0; o < outer; o++) {
                std::memcpy(this->cpuData + o * input1stride * unitSize, old + o * input0stride * unitSize, this->dims[axis] * inner * unitSize);
            }
        } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            uint8_t *old = (uint8_t *)this->cudaData;
            this->MallocSpace(dims[0] * this->stride[0]);
            int outer = this->Count(0) / this->Count(axis);
            int input1stride = this->Count(axis);
            int unitSize = this->unitSize;
            int inner = this->stride[axis];
            FastllmCudaMemcpy2DDeviceToDevice(
                this->cudaData, input1stride * unitSize, old, input0stride * unitSize, this->dims[axis] * inner * unitSize, outer);
            FastllmCudaFree(old);
            FastllmCudaClearBigBuffer();
#else
            ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
        }

    } else {
        this->MallocSpace(dims[0] * this->strides[0]);
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
#    ifdef USE_MMAP
            cpuData = new uint8_t[expansionBytes];
            memcpy(cpuData, this->cpuData, expansionBytes);
#    endif
            // FastllmCudaSetDevice(deviceIds.size() == 0 ? 0 : deviceIds[0]);
            this->cudaData = FastllmCudaMalloc(expansionBytes);
            FastllmCudaCopyFromHostToDevice(this->cudaData, cpuData, expansionBytes);
#    ifdef USE_MMAP
            delete[] cpuData;
#    else
            delete[] this->cpuData;
            this->cpuData = nullptr;
#    endif
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
            a[i] = g_bf16tofp16.dict[b[i]];
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
        int bit = 2;
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

void Data::CreateFromFastllmFormat(uint8_t *datas, uint64_t len) {
    ByteReader reader(datas);
    int version = reader.ReadInt();
    this->dataType = (DataType)reader.ReadInt();
    this->UpdateUnitSize();
    this->Allocate();
    if (version == 1) {
        if (this->dataType == DataType::FP8_E4M3) {
            this->blockK = reader.ReadInt();
            this->blockM = reader.ReadInt();
            this->scales.resize(reader.ReadInt());
            reader.ReadBytes((uint8_t *)this->scales.data(), this->scales.size() * sizeof(float));
            reader.ReadBytes(this->cpuData, len);
        } else if (this->dataType == DataType::INT8 || this->dataType == DataType::INT4 || this->dataType == DataType::INT4_NOZERO) {
            this->perChannelAxis = reader.ReadInt();
            int k = this->perChannelAxis == -1 ? 1 : this->dims[perChannelAxis];
            this->scales.resize(k);
            this->mins.resize(k);
            this->zeros.resize(k);
            for (int i = 0; i < k; i++) {
                if (this->dataType == DataType::INT4_NOZERO) {
                    float min = reader.ReadFloat();
                    float scale = reader.ReadFloat();

                    this->perChannelsConfigs[i] = LowBitConfig(min, min + 15 * scale, 1, 4);
                    this->perChannelsConfigs[i].min = min;
                    this->perChannelsConfigs[i].scale = scale;

                    this->mins[i] = min;
                    this->scales[i] = scale;
                    this->zeros[i] = this->perChannelsConfigs[i].zeroPoint;
                } else if (this->dataType == DataType::INT8 || this->dataType == DataType::INT4) {
                    int bit = this->dataType == DataType::INT4 ? 4 : 8;

                    float min = reader.ReadFloat();
                    float max = reader.ReadFloat();

                    this->perChannelsConfigs[i] = LowBitConfig(min, max, 0, bit);
                    this->mins[i] = min;
                    this->scales[i] = this->perChannelsConfigs[i].scale;
                    this->zeros[i] = this->perChannelsConfigs[i].zeroPoint;

                } else {
                }
            }

            reader.ReadBytes(this->cpuData, len);
        } else if (this->dataType == DataType::INT4_GROUP) {
            this->perChannelAxis = reader.ReadInt();
            this->group = reader.ReadInt();
            this->groupCnt = reader.ReadInt();
            int k = this->perChannelAxis == -1 ? 1 : this->dims[perChannelAxis];

            this->mins.resize(k * this->group);
            this->scales.resize(k * this->groupCnt);

            for (int i = 0; i < k * this->group; i++) {
                this->mins[i] = reader.ReadFloat();
                this->scales[i] = reader.ReadFloat();
            }

            reader.ReadBytes(this->cpuData, len);
        } else {
            ErrorInFastLLM("CreateFromFastllmFormat Error: data type error.");
        }
    } else {
        ErrorInFastLLM("CreateFromFastllmFormat error: unsupport version " + std::to_string(version));
    }
}

void Data::CalcWeightSum() {
    if (this->weightSum.size() > 0) {
        return;
    }
    int n = this->dims[0], m = this->dims[1];
    if (this->dataType == DataType::INT8) {
        weightSum.resize(n);
        std::fill(weightSum.begin(), weightSum.end(), 0);
        for (int i = 0; i < n; i++) {
            int j = 0;
#ifdef __AVX2__
            __m256i acc = _mm256_setzero_si256();
            const __m256i ones = _mm256_set1_epi16(1);
            for (; j + 31 < m; j += 32) {
                __m256i ax = _mm256_loadu_si256((const __m256i *)(cpuData + i * m + j));
                __m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ax, 0));
                __m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ax, 1));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, ones));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, ones));
            }
            weightSum[i] += I32sum(acc);
#endif
#ifdef __aarch64__
            uint32x4_t sum0 = {0, 0, 0, 0};
            for (; j + 7 < m; j += 8) {
                uint8x8_t ori = vld1_u8(cpuData + (i * m + j));
                uint16x4_t sa = vpaddl_u8(ori);
                sum0 = vaddw_u16(sum0, sa);
            }
            weightSum[i] += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#endif
            for (; j < m; j++) {
                weightSum[i] += cpuData[i * m + j];
            }
        }
    } else if (this->dataType == DataType::INT4 || this->dataType == DataType::INT4_NOZERO) {
        weightSum.resize(n);
        std::fill(weightSum.begin(), weightSum.end(), 0);
        for (int i = 0; i < n; i++) {
            int j = 0;
#ifdef __aarch64__
            uint8x8_t maskHigh = vdup_n_u8(0xF0);
            uint8x8_t maskLow = vdup_n_u8(0xF);
            uint32x4_t sum0 = {0, 0, 0, 0};

            for (; j + 15 < m; j += 16) {
                uint8x8_t ori = vld1_u8(cpuData + (i * m + j) / 2);
                uint8x8_t va = vand_u8(ori, maskLow);
                uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);

                uint16x4_t sa = vpaddl_u8(va);
                uint16x4_t sb = vpaddl_u8(vb);

                sum0 = vaddw_u16(sum0, vadd_u16(sa, sb));
            }
            weightSum[i] += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#endif
#ifdef __AVX2__
            __m256i acc = _mm256_setzero_si256();
            const __m256i lowMask = _mm256_set1_epi8(0xf);
            const __m256i ones = _mm256_set1_epi16(1);
            for (; j + 31 < m; j += 32) {
                __m128i orix = _mm_loadu_si128((const __m128i *)(cpuData + (i * m + j) / 2));
                __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                __m256i bx = _mm256_and_si256(lowMask, bytex);

                __m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 0));
                __m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 1));

                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, ones));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, ones));
            }
            weightSum[i] += I32sum(acc);
#endif
            for (; j + 1 < m; j += 2) {
                int id = (i * m + j) / 2;
                weightSum[i] += (cpuData[id] & 0xF) + (cpuData[id] >> 4);
            }
            for (; j < m; j++) {
                int id = (i * m + j) / 2;
                if ((i * m + j) % 2) {
                    weightSum[i] += (cpuData[id] & 0xF);
                } else {
                    weightSum[i] += (cpuData[id] >> 4);
                }
            }
        }
    } else if (this->dataType == DataType::INT4_GROUP) {
        weightSum.resize(n * this->group);
        std::fill(weightSum.begin(), weightSum.end(), 0);
        for (int i = 0; i < n; i++) {
            for (int g = 0; g < this->group; g++) {
                int gid = i * this->group + g;
                int st = g * this->groupCnt;
                int end = std::min(m, (g + 1) * this->groupCnt);
                int j = st;
#ifdef __aarch64__
                uint8x8_t maskHigh = vdup_n_u8(0xF0);
                uint8x8_t maskLow = vdup_n_u8(0xF);
                uint32x4_t sum0 = {0, 0, 0, 0};

                for (; j + 15 < end; j += 16) {
                    uint8x8_t ori = vld1_u8(cpuData + (i * m + j) / 2);
                    uint8x8_t va = vand_u8(ori, maskLow);
                    uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);

                    uint16x4_t sa = vpaddl_u8(va);
                    uint16x4_t sb = vpaddl_u8(vb);

                    sum0 = vaddw_u16(sum0, vadd_u16(sa, sb));
                }
                weightSum[gid] += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#endif
#ifdef __AVX2__
                __m256i acc = _mm256_setzero_si256();
                const __m256i lowMask = _mm256_set1_epi8(0xf);
                const __m256i ones = _mm256_set1_epi16(1);
                for (; j + 31 < end; j += 32) {
                    __m128i orix = _mm_loadu_si128((const __m128i *)(cpuData + (i * m + j) / 2));
                    __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                    __m256i bx = _mm256_and_si256(lowMask, bytex);

                    __m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 0));
                    __m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 1));

                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, ones));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, ones));
                }
                weightSum[gid] += I32sum(acc);
#endif
                for (; j + 1 < end; j += 2) {
                    int id = (i * m + j) / 2;
                    weightSum[gid] += (cpuData[id] & 0xF) + (cpuData[id] >> 4);
                }
                for (; j < end; j++) {
                    int id = (i * m + j) / 2;
                    if ((i * m + j) % 2) {
                        weightSum[gid] += (cpuData[id] & 0xF);
                    } else {
                        weightSum[gid] += (cpuData[id] >> 4);
                    }
                }
            }
        }
    }
}

void Data::Reshape(const std::vector<int> &dims) {
    if (this->dims == dims) {
        return;
    }

    // 计算旧数据的元素总数
    int oldCount = 1;
    for (int d : this->dims) {
        oldCount *= d;
    }

    std::vector<int> newDims = dims;
    int inferIndex = -1;
    long long knownProduct = 1;

    // 检查维度合法性并计算固定维度的乘积
    for (int i = 0; i < newDims.size(); ++i) {
        int d = newDims[i];
        if (d == -1) {
            AssertInFastLLM(inferIndex == -1, "Reshape Error: multiple -1 dimensions are not allowed.");
            inferIndex = i;
        } else {
            AssertInFastLLM(d > 0, "Reshape Error: all dimensions must be positive or -1.");
            knownProduct *= d;
        }
    }

    // 推断 -1 维度
    if (inferIndex != -1) {
        AssertInFastLLM(oldCount % knownProduct == 0, "Reshape Error: total element count not divisible for -1 inference.");
        newDims[inferIndex] = oldCount / knownProduct;
    }

    // 检查新旧元素数量是否匹配
    long long newCount = 1;
    for (int d : newDims) {
        newCount *= d;
    }
    AssertInFastLLM(newCount == oldCount, "Reshape Error: new shape element count mismatch with old tensor.");

    // 执行形状更新
    this->Resize(newDims);
}

void Data::FakeFrom(const Data &ori, size_t offset) {
    this->dataType = ori.dataType;
    this->UpdateUnitSize();
    this->isFake = true;
    this->dataDevice = ori.dataDevice;
    if (this->dataDevice == DataDevice::CPU) {
        this->cpuData = ori.cpuData + offset;
    } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
        this->cudaData = (void *)((uint8_t *)ori.cudaData + offset);
#else
        ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
    }
}

MoEQuantizedExecutor::MoEQuantizedExecutor(Data **weights) {
    this->weights_ = weights;
}

void MoEQuantizedExecutor::prepareBuffer(size_t n, size_t m, size_t group) {
    this->quantizedInput_.clear();
    this->quantizedScales_.clear();
    this->quantizedZeros_.clear();
    this->quantizedLowBitConfigs_.clear();
    this->quantizedSums_.clear();

    this->quantizedInput_.resize(n * m);
    this->quantizedScales_.resize(n * group);
    this->quantizedZeros_.resize(n * group);
    this->quantizedLowBitConfigs_.resize(n * group);
    this->quantizedSums_.resize(n * group);
}

void MoEQuantizedExecutor::ensureMiddleAndResultBuffers(const std::vector<ExpertRoute> &routedExperts) {
    this->middles_.clear();
    this->results.clear();

    this->middles_.resize(routedExperts.size());
    this->results.resize(routedExperts.size());

    for (int i = 0; i < routedExperts.size(); i++) {
        int expertIndex = routedExperts[i].expertIndex;
        Data &weight = *(weights_[expertIndex]);
        this->middles_[i].resize(weight.dims[0]);
        this->results[i].resize(weight.dims[0]);
    }
}

void MoEQuantizedExecutor::ExecuteForOuterIndex(
    int o, float *floatInput, int m, const std::vector<ExpertRoute> &routedExperts, int permuteType) {
    int groupCnt = weights_[2]->groupCnt;
    int group = (m - 1) / groupCnt + 1;
    int n = 1;
    float *curInput = floatInput + o * m;

    prepareBuffer(n, m, group);

    OnlineQuantization(curInput, quantizedInput_, quantizedLowBitConfigs_, 1, m, group, groupCnt, quantizedSums_, quantizedScales_,
        quantizedZeros_, permuteType);

    ensureMiddleAndResultBuffers(routedExperts);

    std::vector<MultiThreadBaseOp *> ops;
    auto *pool = GetAlivePool();
    int thread_nums = pool->threads.size();
    ops.resize(thread_nums);

    for (auto it = routedExperts.begin(); it != routedExperts.end();) {
        auto beginIt = it;
        auto endIt = it;
        const ExpertRoute &expert = *it;
        const Data &expertWeight = *(weights_[2 * expert.expertIndex]);
        const int k = expertWeight.dims[0];

        int selSum = 1;
        int tempSelSum = selSum;
        for (auto jt = std::next(it); jt != routedExperts.end(); jt++) {
            const ExpertRoute &nextExpert = *jt;
            const Data &nextExpertWeight = *(weights_[2 * nextExpert.expertIndex]);
            const int curk = nextExpertWeight.dims[0];

            const int remainder = curk % k;
            if (remainder != 0) {
                break;
            }

            int tempSelSum = tempSelSum + curk / k;
            if (thread_nums % tempSelSum == 0) {
                selSum = tempSelSum;
                endIt = jt;
            }
        }

        const int base = thread_nums / selSum;
        int threadSt = 0;

        for (auto jt = beginIt; jt != std::next(endIt); jt++) {
            const ExpertRoute &expert = *jt;
            Data &upWeight = *(weights_[2 * expert.expertIndex]);
            const int curk = upWeight.dims[0];
            int curThread = (curk / k) * base;
            const int index = jt - routedExperts.begin();
            std::vector<float> &middle = this->middles_[index];

            upWeight.CalcWeightSum();

            n = 1;
            if (upWeight.dataType == DataType::INT8) {
                LaunchLinearInt8Int8(quantizedInput_.data(), upWeight.cpuData, middle.data(), n, m, curk, upWeight.weightSum.data(),
                    upWeight.zeros.data(), upWeight.scales.data(), nullptr, quantizedSums_.data(), quantizedScales_.data(),
                    quantizedZeros_.data(), ops, pool, threadSt, curThread);
            } else {
                MultiplyInt4GroupMultiThreadLaunch(quantizedInput_.data(), upWeight.cpuData, middle.data(), n, m, curk,
                    upWeight.weightSum.data(), upWeight.mins.data(), upWeight.scales.data(), nullptr, quantizedSums_, quantizedScales_,
                    quantizedZeros_, quantizedLowBitConfigs_, threadSt, curThread, group, groupCnt, ops, pool);
            }

            threadSt += curThread;
        }

        for (int j = 0; j < ops.size(); j++) {
            pool->Wait(j);
            delete ops[j];
        }

        for (auto jt = beginIt; jt != std::next(endIt); jt++) {
            const ExpertRoute &expert = *jt;
            Data &upWeight = *(weights_[2 * expert.expertIndex]);
            const int curk = upWeight.dims[0];
            const int index = jt - routedExperts.begin();
            std::vector<float> &middle = this->middles_[index];
            int mid = curk / 2;

            ops[jt - beginIt] = new MultiThreadMultiOps();
            ((fastllm::MultiThreadMultiOps *)ops[l - st])
                ->ops.push_back(new fastllm::MultiThreadSwigluOp(outputData, mid, mid, outputData, 1, spatial, spatial));

            n = 1;
            new MultiThreadSwigluOp(middle.data(), mid, mid, middle.data(), n, curk, curk);

            Data &downWeight = *(weights_[2 * expert.expertIndex + 1]);
            groupCnt = downWeight.groupCnt;
            group = (mid - 1) / groupCnt + 1;

            quantizedMiddleScales_.clear();
            quantizedMiddleZeros_.clear();
            quantizedMiddleInput_.clear();
            quantizedMiddleLowBitConfigs_.clear();
            quantizedMiddleSums_.clear();

            quantizedMiddleScales_[index].resize(n * group);
            quantizedMiddleZeros_[index].resize(n * group);
            quantizedMiddleInput_[index].resize(n * mid);
            quantizedMiddleLowBitConfigs_[index].resize(n * group);
            quantizedMiddleSums_[index].resize(n * group);

            new MultiThreadOnlineQuantizationOp(middle.data(), quantizedMiddleInput_[index].data(), quantizedMiddleLowBitConfigs_[index].data(),
                n, mid, group, groupCnt, quantizedMiddleSums_[index].data(), quantizedMiddleScales_[index].data(),
                quantizedMiddleZeros_[index].data(), permuteType);
        }

        it = std::next(endIt);
    }
}
