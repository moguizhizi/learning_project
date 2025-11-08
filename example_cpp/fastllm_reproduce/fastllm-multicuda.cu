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

bool SplitMultiCudaWeight(Data &weight, Data &bias, std::vector<int> &multiCudaCurrentDevices, DivisionScheme divisionScheme, int splitAxis) {}