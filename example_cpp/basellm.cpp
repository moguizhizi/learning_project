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
        this->cpudata = (uint8_t *)ptr;
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
        std::memcpy(this->cpudata, data.data(), this->GetBytes());
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
    delete[] this->cpudata;
    delete[] this->cudadata;

    this->cpudata = nullptr;
    this->cudadata = nullptr;
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
        this->cpudata = new uint8_t[this->expansionBytes];
        std::memset(this->cpudata, 0, this->expansionBytes * sizeof(uint8_t));
    }
}