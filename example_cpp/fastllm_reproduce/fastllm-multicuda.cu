#include "fastllm-cuda.cuh"
#include "fastllm-multicuda.cuh"

std::map<int, std::string> specialDeviceIds = {{99999, "cpu"}};

void SwitchDeviceAndGetInfos(int deviceId, std::string &specialId, int &mallocType) {
    specialId = "";
    if (specialDeviceIds.find(deviceId) == specialDeviceIds.end()) {
        cudaSetDevice(deviceId);
    } else {
        specialId = specialDeviceIds[deviceId];
    }
    mallocType = 1;
    if (specialId == "cpu") {
        mallocType = 0;
    }
}

void CopyToMultiDevices(Data &data, std::vector<int> devices, bool copyData) {
    if (data.multiDeviceData) {
        return;
    }

    data.multiDeviceData = true;

    int orid = FastllmCudaGetDevice();
    if (copyData) {
        data.ToDevice(DataDevice::CPU);
        for (auto &device : devices) {
            std::string specialId;
            int mallocType = 0;
            SwitchDeviceAndGetInfos(device, specialId, mallocType);
            DataDevice datadevice = (mallocType == 0 ? DataDevice::CPU : DataDevice::CUDA);

            data.multiDeviceDatas[device] = new Data();
            data.multiDeviceDatas[device]->CopyFrom(data);
            data.multiDeviceDatas[device]->ToDevice(datadevice);

            data.multiDeviceDatas[device]->group = data.group;
            data.multiDeviceDatas[device]->groupCnt = data.groupCnt;
            data.multiDeviceDatas[device]->scales = data.scales;
            data.multiDeviceDatas[device]->mins = data.mins;
            data.multiDeviceDatas[device]->zeros = data.zeros;
            data.multiDeviceDatas[device]->halfScales = data.halfScales;
        }
    } else {
        for (auto &device : devices) {
            std::string specialId;
            int mallocType = 0;
            SwitchDeviceAndGetInfos(device, specialId, mallocType);
            DataDevice datadevice = (mallocType == 0 ? DataDevice::CPU : DataDevice::CUDA);

            if (data.dims.size() == 0) {
                data.multiDeviceDatas[device] = new Data(data.dataType);
            } else {
                data.multiDeviceDatas[device] = new Data(data.dataType, data.dims);
            }
            data.multiDeviceDatas[device]->dataDevice = datadevice;
        }
    }
    FastllmCudaSetDevice(orid);
}

void *AutoMalloc(size_t size, int type) {
    if (type == 0) {
        return (void *)(new uint8_t[size]);
    } else {
        return (void *)FastllmCudaMalloc(size);
    }
}

cudaError_t AutoMemset(void *a, int value, size_t size, int type) {
    if (type == 0) {
        memset(a, value, size);
        return cudaSuccess;
    } else {
        return cudaMemset(a, value, size);
    }
}

cudaMemcpyKind GetCudaMemcpyType(int dstType, int srcType) {
    if (srcType == 0) {
        if (dstType == 0) {
            return cudaMemcpyHostToHost;
        } else {
            return cudaMemcpyHostToDevice;
        }
    } else {
        if (dstType == 0) {
            return cudaMemcpyDeviceToHost;
        } else {
            return cudaMemcpyDeviceToDevice;
        }
    }
}

std::vector<int> multiCudaCurrentDevices;
std::map<int, int> multiCudaCurrentRatios;

void FastllmMultiCudaSetDevice(std::vector<int> ids) {
    multiCudaCurrentDevices = ids;
}

void FastllmMultiCudaSetDeviceRatio(std::map<int, int> &deviceRatio) {
    multiCudaCurrentRatios = deviceRatio;
}

void FastllmGetMulticudaDeviceAndRatio(std::vector<int> &devices, std::map<int, int> &ratios, bool noSpecial) {
    devices.clear();
    ratios.clear();
    for (int i : multiCudaCurrentDevices) {
        if (noSpecial == false || specialDeviceIds.find(i) == specialDeviceIds.end()) {
            devices.push_back(i);
            ratios[i] = multiCudaCurrentRatios.find(i) != multiCudaCurrentRatios.end() ? multiCudaCurrentRatios[i] : 1;
        }
    }
}

// 将total个计算任务切分
// 若当前有x个设备，返回一个长度为(x + 1)的vector，第i个设备执行任务[ret[i], ret[i + 1])
std::vector<int> FastllmMultiCudaGetSplitPoints(
    std::vector<int> &multiCudaCurrentDevices, std::map<int, int> &multiCudaCurrentRatios, int total, int unit = 1) {
    int deviceNum = multiCudaCurrentDevices.size();
    int nodes = total / unit;
    int totalRatio = 0;
    if (multiCudaCurrentRatios.size() > 0) {
        for (auto &it : multiCudaCurrentRatios) {
            totalRatio += it.second;
        }
    } else {
        totalRatio = deviceNum;
    }
    std::vector<int> ret;
    int cur = 0;
    for (int i = 0; i < deviceNum; i++) {
        int curRatio = 1;
        if (multiCudaCurrentRatios.find(multiCudaCurrentDevices[i]) != multiCudaCurrentRatios.end()) {
            curRatio = multiCudaCurrentRatios[i];
        }
        int now = std::max(1, nodes * curRatio / totalRatio) * unit;
        int end = (i == deviceNum - 1 ? total : cur + now);
        ret.push_back(cur);
        if (i == deviceNum - 1) {
            ret.push_back(end);
        }
        cur = end;
    }
    return ret;
}

std::vector<bool> streamInits = std::vector<bool>(4, 0);
cudaStream_t streams[4];

cudaStream_t *GetFastllmStream(int id) {
    if (!streamInits[id]) {
        streamInits[id] = true;
        cudaSetDevice(id);
        cudaStreamCreate(&streams[id]);
        cudaSetDevice(0);
    }
    return &streams[id];
}

void EnablePeerAccessAll(const std::vector<int> &devices) {
    int curdev = -1;
    auto setDevice = [&](int i) {
        if (curdev != i) {
            cudaSetDevice(i);
            curdev = i;
        }
    };

    for (int i = 0; i < devices.size(); i++) {
        for (int j = i + 1; j < devices.size(); j++) {
            int A = devices[i];
            int B = devices[j];
            int canAB = 0, canBA = 0;

            cudaDeviceCanAccessPeer(&canAB, A, B);
            cudaDeviceCanAccessPeer(&canBA, B, A);

            if (canAB) {
                setDevice(A), cudaDeviceEnablePeerAccess(B, 0);
            }

            if (canBA) {
                setDevice(B), cudaDeviceEnablePeerAccess(A, 0);
            }
        }
    }
}

bool SplitMultiCudaWeight(Data &weight, Data &bias, std::vector<int> &multiCudaCurrentDevices, DivisionScheme divisionScheme, int splitAxis) {
    if (weight.multiDeviceData && bias.multiDeviceData) {
        return true;
    }

    weight.multiDeviceData = true;
    bias.multiDeviceData = true;
    int deviceNum = multiCudaCurrentDevices.size();
    int k = weight.dims[0];
    int m = weight.dims[1];
    DataType weightDataType = weight.dataType;
    DataType biasDataType = bias.dataType;
    uint8_t *devWeightData = nullptr;
    uint8_t *devBiasData = nullptr;
    int elementSize = weight.unitSize / weight.unitSizeDiv;

    FastllmCudaSetDevice(0);

    float *biasCuda = (float *)FastllmCudaMalloc(k * sizeof(float));
    if (bias.dims.size() > 0) {
        cudaMemcpy((uint8_t *)biasCuda, (uint8_t *)bias.cudaData, k * sizeof(float), GetCudaMemcpyType(1, 1));
    } else {
        cudaMemset((uint8_t *)biasCuda, 0, k * sizeof(float));
    }
    EnablePeerAccessAll(multiCudaCurrentDevices);

    for (int i = 0; i < deviceNum; i++) {
        int deviceID = multiCudaCurrentDevices[i];

        // --- 1. 切换设备并获取信息 ---
        int mallocType = 0;
        std::string specialId;
        SwitchDeviceAndGetInfos(deviceID, specialId, mallocType);
        DataDevice dataDevice = (mallocType == 0 ? DataDevice::CPU : DataDevice::CUDA);

        // --- 2. 计算当前设备分得的长度 ---
        const auto &devScheme = divisionScheme[deviceID];
        int len = 0;
        for (const auto &it : devScheme) len += (it.second - it.first);

        // --- 3. 创建子 weight / bias 张量 ---
        std::vector<int> devWeightDims, devBiasDims;
        if (splitAxis == 0) {
            devWeightDims = {len, m};
            devBiasDims = {len};
        } else {
            devWeightDims = {k, len};
            devBiasDims = {k};
        }

        weight.multiDeviceDatas[deviceID] = new Data(weightDataType, devWeightDims);
        bias.multiDeviceDatas[deviceID] = new Data(biasDataType, devBiasDims);

        Data *devWeight = weight.multiDeviceDatas[deviceID];
        Data *devBias = bias.multiDeviceDatas[deviceID];

        devWeight->dataDevice = dataDevice;
        devBias->dataDevice = dataDevice;

        uint8_t *devWeightData = (uint8_t *)(mallocType == 0 ? devWeight->cpuData : devWeight->cudaData);
        uint8_t *devBiasData = (uint8_t *)(mallocType == 0 ? devBias->cpuData : devBias->cudaData);

        // --- 4. 根据 splitAxis 拷贝数据 ---
        int curLen = 0;
        for (const auto &[start, end] : devScheme) {
            int sliceLen = end - start;

            if (splitAxis == 0) {
                // 行分块：拷贝连续区间 [it.first : it.second)
                cudaMemcpy(devWeightData + curLen * m * elementSize, (uint8_t *)weight.cudaData + start * m * elementSize,
                    sliceLen * m * elementSize, GetCudaMemcpyType(mallocType, 1));

                cudaMemcpy(devBiasData + curLen * sizeof(float), (uint8_t *)bias.cudaData + start * sizeof(float), sliceLen * sizeof(float),
                    GetCudaMemcpyType(mallocType, 1));
            } else {
                // 列分块：使用 2D 拷贝
                FastllmCudaMemcpy2D(devWeightData + curLen * elementSize, len * elementSize, (uint8_t *)weight.cudaData + start * elementSize,
                    m * elementSize,
                    sliceLen * elementSize, // width
                    k,                      // height
                    GetCudaMemcpyType(mallocType, 1), deviceID, 0);
            }

            curLen += sliceLen;
        }

        // --- 5. 处理 bias ---
        if (splitAxis == 1) {
            if (i == 0) {
                cudaMemcpy(devBiasData, (uint8_t *)biasCuda, k * sizeof(float), GetCudaMemcpyType(1, 1));
            } else {
                AutoMemset(devBiasData, 0, k * sizeof(float), mallocType);
            }
        }
    }

    int weightGroup = weight.group = -1 ? 1 : weight.group;
    int weightGroupCnt = weight.groupCnt;
    const std::vector<float> &weightScales = weight.scales;
    const std::vector<float> &weightMins = weight.mins;
    std::vector<int> weightZeros;

    weightZeros.resize(k * weightGroup);
    if (weight.perChannelsConfigs.size() > 0) {
        for (int i = 0; i < k * weightGroup; i++) {
            weightZeros[i] = weight.perChannelsConfigs[i].zeroPoint;
        }
    } else if (weight.zeros.size() > 0) {
        weightZeros = weight.zeros;
    } else {
        std::fill(weightZeros.begin(), weightZeros.end(), 0);
    }

    if (weight.dataType == DataType::FP8_E4M3) {
    } else if (weight.mins.size() > 0) {
        for (int i = 0; i < deviceNum; i++) {
            int deviceID = multiCudaCurrentDevices[i];
            const auto &devScheme = divisionScheme[deviceID];
            int len = 0;
            for (const auto &it : devScheme) len += (it.second - it.first);

            Data *devWeight = weight.multiDeviceDatas[deviceID];

            devWeight->group = weight.group;
            devWeight->groupCnt = weight.groupCnt;

            int devScaleCount = 0;
            int devMinsCount = 0;
            int devZerosCount = 0;
            if (splitAxis == 0) {
                devScaleCount = len * weightGroup;
                devMinsCount = len * weightGroup;
                devZerosCount = len * weightGroup;
            } else {
                devScaleCount = k * weightGroup;
                devMinsCount = k * weightGroup;
                devZerosCount = k * weightGroup;
            }

            std::vector<float> &devScales = devWeight->scales;
            std::vector<float> &devMins = devWeight->mins;
            std::vector<int> &devZeros = devWeight->zeros;

            devScales.resize(devScaleCount, 0.0f);
            devMins.resize(devMinsCount, 0.0f);
            devZeros.resize(devZerosCount, 0);

            int curLen = 0;
            int curGroupLen = 0;
            for (const auto &[start, end] : devScheme) {
                int sliceLen = end - start;

                if (splitAxis == 0) {
                    memcpy(devScales.data() + curLen, weightScales.data() + start * weightGroup, sliceLen * weightGroup * sizeof(float));
                    memcpy(devMins.data() + curLen, weightMins.data() + start * weightGroup, sliceLen * weightGroup * sizeof(float));
                    memcpy(devZeros.data() + curLen, weightZeros.data() + start * weightGroup, sliceLen * weightGroup * sizeof(int));
                } else {
                    if (weightGroupCnt == -1) {
                        // 情况1：未分组，直接整体拷贝
                        memcpy(devScales.data(), weightScales.data(), k * sizeof(float));
                        memcpy(devMins.data(), weightMins.data(), k * sizeof(float));
                        memcpy(devZeros.data(), weightZeros.data(), k * sizeof(int));
                    } else {
                        // 情况2：按分组进行拷贝
                        const bool startAligned = (start % weightGroupCnt == 0);
                        const bool endAligned = (end % weightGroupCnt == 0);

                        const int startGroup = start / weightGroupCnt;
                        const int endGroup = end / weightGroupCnt;
                        const int sliceGroupLen = endGroup - startGroup;

                        if (startAligned && endAligned) {
                            // 对齐时整块拷贝，每行的 group 数据连续
                            for (int row = 0; row < k; ++row) {
                                const float *srcScale = weightScales.data() + row * weightGroup + startGroup;
                                const float *srcMin = weightMins.data() + row * weightGroup + startGroup;
                                const int *srcZero = weightZeros.data() + row * weightGroup + startGroup;

                                float *dstScale = devScales.data() + row * weightGroup + curGroupLen;
                                float *dstMin = devMins.data() + row * weightGroup + curGroupLen;
                                int *dstZero = devZeros.data() + row * weightGroup + curGroupLen;

                                const size_t copyF = sliceGroupLen * sizeof(float);
                                const size_t copyI = sliceGroupLen * sizeof(int);

                                memcpy(dstScale, srcScale, copyF);
                                memcpy(dstMin, srcMin, copyF);
                                memcpy(dstZero, srcZero, copyI);
                            }

                            curGroupLen += sliceGroupLen;
                        } else {
                            // TODO: 非对齐情况处理（如果有）
                            // printf("Warning: group range [%d, %d) not aligned with weightGroupCnt=%d\n", start, end, weightGroupCnt);
                        }
                    }
                }
                curLen += sliceLen;
            }
        }
    }