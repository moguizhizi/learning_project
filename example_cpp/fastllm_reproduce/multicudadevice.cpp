#include "multicudadevice.h"

#include "fastllm-cuda.cuh"
#include "fastllm-multicuda.cuh"

MultiCudaDevice::MultiCudaDevice(CudaDevice *cudaDevice) {
    this->cudaDevice = cudaDevice;
    this->deviceType = "multicuda";

    this->ops["MLP"] = (BaseOperator *)(new MultiCudaMLPOp());
    this->ops["Linear"] = (BaseOperator *)(new MultiCudaLinearOp());
    this->ops["MergeMOE"] = (BaseOperator *)(new MultiCudaMergeMOE());
    this->ops["MergeAttention"] = (BaseOperator *)(new MultiCudaMergeAttention());
}

bool MultiCudaDevice::Malloc(void **ret, size_t size) {
    *ret = FastllmCudaMalloc(size);
    return true;
}

bool MultiCudaDevice::Free(void *ret) {
    FastllmCudaFree(ret);
    return true;
}

bool MultiCudaDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
    FastllmCudaCopyFromHostToDevice(dst, src, size);
    return true;
}

bool MultiCudaDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
    FastllmCudaCopyFromDeviceToHost(dst, src, size);
    return true;
}

bool MultiCudaDevice::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    if (this->ops.find(opType) == this->ops.end()) {
        if (((BaseDevice *)this->cudaDevice)->ops.find(opType) == ((BaseDevice *)this->cudaDevice)->ops.end()) {
            return false;
        } else {
            return ((BaseDevice *)this->cudaDevice)->CanRun(opType, datas, floatParams, intParams);
        }
    } else {
        return this->ops[opType]->CanRun(opType, datas, floatParams, intParams);
    }
}

// 对某一个算子进行形状推理
void MultiCudaDevice::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    if (this->ops.find(opType) == this->ops.end()) {
        ((BaseDevice *)this->cudaDevice)->Reshape(opType, datas, floatParams, intParams);
    } else {
        this->ops[opType]->Reshape(opType, datas, floatParams, intParams);
    }
}

// 对某一个算子进行推理
void MultiCudaDevice::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    if (this->ops.find(opType) == this->ops.end()) {
        ((BaseDevice *)this->cudaDevice)->Run(opType, datas, floatParams, intParams);
    } else {
        this->ops[opType]->Run(opType, datas, floatParams, intParams);
    }
}

bool MultiCudaLinearOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    if (intParams.find("exType") != intParams.end()) {
        return false;
    }
    Data &weight = *(datas.find("weight")->second);
    return weight.dims[0] > 10000 || weight.dims[1] > 10000;
}

void MultiCudaLinearOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &output = *(datas.find("output")->second);
    Data &weight = *(datas.find("weight")->second);
    Data &bias = *(datas.find("bias")->second);

    int n = input.Count(0) / input.dims.back();
    int m = input.dims.back();
    int k = output.dims.back();

    int groupCnt = weight.groupCnt;
    int blockK = weight.blockK;
    DataType dataType = weight.dataType;

    std::vector<int> devices;
    std::map<int, int> ratios;

    output.Allocate();

    int uint = (groupCnt = -1 ? 128 : groupCnt);
    if (dataType == DataType::FP8_E4M3) {
        uint = blockK;
    }

    FastllmGetMulticudaDeviceAndRatio(devices, ratios, false);
    std::vector<int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, k, uint);

    DivisionScheme divisionScheme;
    for (int i = 0; i < devices.size(); i++) {
        divisionScheme[devices[i]].push_back(std::make_pair(points[i], points[i + 1]));
    }

    SplitMultiCudaWeight(weight, bias, devices, divisionScheme, 0);
    CopyToMultiDevices(input, devices, false);
    CopyToMultiDevices(output, devices, false);

    auto *pool = GetAlivePool();
    std::vector<MultiThreadBaseOp *> ops;
    for (int i = 0; i < devices.size(); i++) {
        int deviceId = devices[i];
        int start = points[i];
        int len = points[i + 1] - points[i];
        ops.push_back(
            new MultiCudaDoLinearOp((uint8_t *)input.cudaData, nullptr, input.multiDeviceDatas[deviceId], weight.multiDeviceDatas[deviceId],
                bias.multiDeviceDatas[deviceId], output.multiDeviceDatas[deviceId], n, m, k, start, len, (uint8_t *)output.cudaData, deviceId));
    }

    for (int i = 0; i < devices.size(); i++) {
        pool->PushOp(i, ops[i]);
    }
    for (int i = 0; i < devices.size(); i++) {
        pool->Wait(i);
        delete ops[i];
    }
}

void MultiCudaMergeAttention::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &weight1 = *(datas.find("weight1")->second);
    Data &output = *(datas.find("output")->second);
    std::vector<int> dims = input.dims;
    dims.back() = weight1.dims[0];
    output.dataType = input.dataType;
    output.Resize(dims);
}

void MultiCudaMergeAttention::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
    Data &input = *(datas.find("input")->second);
    Data &weight0 = *(datas.find("weight0")->second);
    Data &bias0 = *(datas.find("bias0")->second);
    Data &weight1 = *(datas.find("weight1")->second);
    Data &bias1 = *(datas.find("bias1")->second);
    Data &positionIds = *(datas.find("positionIds")->second);
    Data &sinData = *(datas.find("sinData")->second);
    Data &cosData = *(datas.find("cosData")->second);
    Data &output = *(datas.find("output")->second);
    Data &qkv = *(datas.find("qkv")->second);
    Data &q = *(datas.find("q")->second);
    Data &k = *(datas.find("k")->second);
    Data &v = *(datas.find("v")->second);
    int qNum = intParams.find("qNum")->second;
    int kvNum = intParams.find("kvNum")->second;
    int headDim = intParams.find("headDim")->second;
    int rotDim = intParams.find("rotDim")->second;
    float attentionScale = floatParams.find("attentionScale")->second;
    Data **keys = (Data **)(datas.find("keys")->second);
    Data **values = (Data **)(datas.find("values")->second);
    Data **masks = (Data **)(datas.find("masks")->second);

    int batch = intParams.find("keys___batch")->second;
}