#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Device Name:                             %s\n",  devProp.name);
    printf("CUDA Arch Major revision number:         %d\n",  devProp.major);
    printf("CUDA Arch Minor revision number:         %d\n",  devProp.minor);
    printf("Total global memory (MB):                %u\n",  devProp.totalGlobalMem / (size_t)(1024.0 * 1024.0));
    printf("Total shared memory per block (KB):      %u\n",  devProp.sharedMemPerBlock / (size_t)1024.0);
    printf("Total registers per block:               %d\n",  devProp.regsPerBlock);
    printf("L2 Cach size (KB):                       %d\n",  devProp.l2CacheSize / (size_t)1024.0);
    printf("Total constant memory (KB):              %u\n",  devProp.totalConstMem / (size_t)1024.0);
    printf("Maximum memory pitch:                    %u\n",  devProp.memPitch);
    printf("Processor clock rate (MHz):              %d\n",  devProp.clockRate / (size_t)1000.0);
    printf("Memory clock rate (MHz):                 %u\n",  devProp.memoryClockRate / (size_t)1000.0);
    printf("Memory Bus Width (bits):                 %u\n",  devProp.memoryBusWidth);
    printf("Number of multiprocessors:               %d\n",  devProp.multiProcessorCount);
    printf("Warp size:                               %d\n",  devProp.warpSize);
    printf("Maximum threads per multiprocessors:     %d\n",  devProp.maxThreadsPerMultiProcessor);
    printf("Maximum threads per block:               %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:            %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:             %d\n", i, devProp.maxGridSize[i]);
    printf("Texture alignment:                       %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution:           %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Kernel execution timeout:                %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    printf("ECC enabled:                             %s\n",  (devProp.ECCEnabled ? "Yes" : "No"));
    return;
}

int main()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }

    //getchar();

    return 0;
}
