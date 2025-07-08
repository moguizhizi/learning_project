#include "basellm.h"
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

Data::Data(DataType datatype, const std::vector<int> &dims, DataDevice device, const std::vector<float> data) : Data::Data(datatype, dims) {
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
    this->dims.resize(0);
    this->stride.resize(0);
    delete[] this->cpuData;
    delete[] this->cudaData;

    this->cpuData = nullptr;
    this->cudaData = nullptr;
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

void Data::ToDevice(DataDevice device) {}

void Data::CopyFrom(Data &ori) {

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