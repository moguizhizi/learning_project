#include <stdio.h>
#include <cuda.h>

int main() {
    // 1. 初始化 CUDA Driver API
    cuInit(0);

    // 2. 获取第 0 个 GPU 设备
    CUdevice dev;
    cuDeviceGet(&dev, 0);

    // 3. 查询 SM 数量
    int sm_count = 0;
    cuDeviceGetAttribute(
        &sm_count,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        dev
    );

    // 4. 查询 warp 大小
    int warp_size = 0;
    cuDeviceGetAttribute(
        &warp_size,
        CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        dev
    );

    // 5. 查询每个 block 的共享内存
    int shared_mem = 0;
    cuDeviceGetAttribute(
        &shared_mem,
        CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK,
        dev
    );

    // 6. 查询 Compute Capability
    int major = 0, minor = 0;
    cuDeviceGetAttribute(
        &major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        dev
    );
    cuDeviceGetAttribute(
        &minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        dev
    );

    int device_vmm = 0;
    cuDeviceGetAttribute(
        &device_vmm, 
        CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, 
        dev);

    // 7. 打印结果
    printf("SM Count: %d\n", sm_count);
    printf("Warp Size: %d\n", warp_size);
    printf("Shared Mem Per Block: %d KB\n", shared_mem / 1024);
    printf("Compute Capability: %d.%d\n", major, minor);
    printf("VMM Supported: %s\n", device_vmm ? "YES" : "NO");


    return 0;
}
