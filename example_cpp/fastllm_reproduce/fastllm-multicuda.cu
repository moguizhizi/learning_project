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