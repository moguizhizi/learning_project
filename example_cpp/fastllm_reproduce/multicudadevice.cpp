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

    output.Allocate();
    int group = qNum / kvNum;
    int vDim = weight1.dims[0] / qNum;
    std::vector<int> devices;
    std::map<int, int> ratios;
    FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
    std::vector<int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, kvNum, 1);
    DivisionScheme divisionScheme, divisionSchemeO;
    for (int i = 0; i < devices.size(); i++) {
        int st = points[i], end = points[i + 1];
        int deviceId = devices[i];
        int qgap = qNum * headDim, qkgap = (qNum + kvNum) * headDim;
        divisionScheme[deviceId].push_back(std::make_pair(st * group * headDim, end * group * headDim));
        divisionScheme[deviceId].push_back(std::make_pair(qgap + st * headDim, qgap + end * headDim));
        divisionScheme[deviceId].push_back(std::make_pair(qkgap + st * headDim, qkgap + end * headDim));

        divisionSchemeO[deviceId].push_back(std::make_pair(st * group * vDim, end * group * vDim));
    }
    SplitMultiCudaWeight(weight0, bias0, devices, divisionScheme, 0);
    SplitMultiCudaWeight(weight1, bias1, devices, divisionSchemeO, 1);
    CopyToMultiDevices(qkv, devices, false);
    CopyToMultiDevices(q, devices, false);
    CopyToMultiDevices(k, devices, false);
    CopyToMultiDevices(v, devices, false);
    CopyToMultiDevices(positionIds, devices, true);
    CopyToMultiDevices(sinData, devices, true);
    CopyToMultiDevices(cosData, devices, true);
    for (int i = 0; i < batch; i++) {
        CopyToMultiDevices(*keys[i], devices, true);
        CopyToMultiDevices(*values[i], devices, true);
        if (masks[i] != nullptr) {
            CopyToMultiDevices(*masks[i], devices, true);
        }
    }
    std::map<int, std::vector<Data *> > curKeys, curValues, curMasks;
    for (int device : devices) {
        for (int i = 0; i < batch; i++) {
            curKeys[device].push_back(keys[i]->multiDeviceDatas[device]);
            curValues[device].push_back(values[i]->multiDeviceDatas[device]);
            curMasks[device].push_back(masks[i] == nullptr ? nullptr : masks[i]->multiDeviceDatas[device]);
        }
    }

    Data &curInput = *(datas.find("curInput")->second);
    Data &curOutput = *(datas.find("curOutput")->second);

    CopyToMultiDevices(input, devices, false);
    curOutput.dataDevice = input.dataDevice;
    CopyToMultiDevices(curOutput, devices, false);
    std::vector<uint8_t> cpuInput;
    cpuInput.resize(input.GetBytes());
    FastllmCudaSetDevice(0);
    FastllmCudaCopyFromDeviceToHost(cpuInput.data(), input.cudaData, input.GetBytes());
    uint8_t *partOutput = (uint8_t *)FastllmCudaMalloc(output.GetBytes() * devices.size());
    auto *pool = GetAlivePool();
    std::vector<MultiThreadBaseOp *> ops;

    for (int i = 0; i < devices.size(); i++) {
        int device = devices[i];
        FastllmCudaSetDevice(device);
        int bsz = batch, seqlen = input.dims[1];
        if (bsz > 1) {
            seqlen = 1;
        }

        int unitLen = 128;
        for (int i = 0; i < bsz; i++) {
            Data &pastKey = *keys[i]->multiDeviceDatas[device];
            Data &pastValue = *values[i]->multiDeviceDatas[device];
            while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || seqlen > pastKey.expansionDims[1])) ||
                   (pastKey.dims.size() > 0 && pastKey.dims[1] + seqlen > pastKey.expansionDims[1])) {
                std::vector<int> newDims;
                if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector<int>{kvNum, ((seqlen - 1) / unitLen + 1) * unitLen, headDim};
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((seqlen - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }
            while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || seqlen > pastValue.expansionDims[1])) ||
                   (pastValue.dims.size() > 0 && pastValue.dims[1] + seqlen > pastValue.expansionDims[1])) {
                std::vector<int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector<int>{kvNum, ((seqlen - 1) / unitLen + 1) * unitLen, headDim};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((seqlen - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }
        }
    }
    FastllmCudaSetDevice(0);

    for (int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        ops.push_back(new MultiCudaDoMergeAttentionOp((uint8_t *)input.cudaData, (uint8_t *)cpuInput.data(), partOutput + output.GetBytes() * i,
            input.multiDeviceDatas[device], weight0.multiDeviceDatas[device], bias0.multiDeviceDatas[device], weight1.multiDeviceDatas[device],
            bias1.multiDeviceDatas[device], qkv.multiDeviceDatas[device], q.multiDeviceDatas[device], k.multiDeviceDatas[device],
            v.multiDeviceDatas[device], qNum, kvNum, headDim, rotDim, attentionScale, positionIds.multiDeviceDatas[device],
            sinData.multiDeviceDatas[device], cosData.multiDeviceDatas[device], curKeys[device].data(), curValues[device].data(),
            curMasks[device].data(), curOutput.multiDeviceDatas[device], batch, device));
    }
    for (int i = 0; i < devices.size(); i++) {
        pool->PushOp(i, ops[i]);
    }
    for (int i = 0; i < devices.size(); i++) {
        pool->Wait(i);
        delete ops[i];
    }
    FastllmReduce((uint8_t *)output.cudaData, partOutput, output.Count(0), devices.size(), output.dataType);
    FastllmCudaFree(partOutput);
    for (int i = 0; i < batch; i++) {
        keys[i]->dims = keys[i]->multiDeviceDatas[devices[0]]->dims;
        keys[i]->expansionDims = keys[i]->multiDeviceDatas[devices[0]]->expansionDims;
        values[i]->dims = values[i]->multiDeviceDatas[devices[0]]->dims;
        values[i]->expansionDims = values[i]->multiDeviceDatas[devices[0]]->expansionDims;
    }
}