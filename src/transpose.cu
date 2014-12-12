#include <stdio.h>
#include <iostream>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//NVTX Dir: C:\Program Files\NVIDIA GPU Computing Toolkit\nvToolsExt
#include <nvToolsExt.h>

//Initialize sizes
const int rows = 4096;
const int cols = 4096;
const int BLOCK_SIZE_X = 32;
const int BLOCK_SIZE_Y = 32;

//For unrolled transpose
const int TILE = 32;
const int SIDE = 8;

using namespace std;

#define NAIVE_TRANSPOSE      1
#define SHARED_MEM_TRANSPOSE 1
#define BANK_CONF_TRANSPOSE  1
#define UNROLLED_TRANSPOSE   1

struct DIMS
{
    dim3 dimBlock;
    dim3 dimGrid;
};

#define CUDA(call) do {                             \
    cudaError_t e = (call);                         \
    if (e == cudaSuccess) break;                    \
    fprintf(stderr, __FILE__":%d: %s (%d)\n",       \
            __LINE__, cudaGetErrorString(e), e);    \
    exit(1);                                        \
} while (0)

double diffclock( clock_t clock1, clock_t clock2 )
{
    double diffticks = clock1 - clock2;
    double diffms    = diffticks / ( CLOCKS_PER_SEC / 1000.0);
    return diffms;
}

inline unsigned divup(unsigned n, unsigned div)
{
    return (n + div - 1) / div;
}

// Check errors
void postprocess(const float *ref, const float *res, int n)
{
    bool passed = true;
    for (int i = 0; i < n; i++)
    {
        if (res[i] != ref[i])
        {
            printf("ID:%d \t Res:%f \t Ref:%f\n", i, res[i], ref[i]);
            printf("%25s\n", "*** FAILED ***");
            passed = false;
            break;
        }
    }
    if(passed)
        printf("Post process check passed!!\n");
}

void preprocess(float *res, float *dev_res, int n)
{
    for (int i = 0; i < n; i++)
    {
        res[i] = -1;
    }
    cudaMemset(dev_res, -1, n * sizeof(float));
}

__global__ void copyKernel(const float* __restrict__ const a,
        float* __restrict__ const b)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col

    int index_in = i*cols+j;   // (i,j) from matrix A

    b[index_in] = a[index_in];
}

__global__ void matrixTransposeNaive(const float* __restrict__ const a,
        float* __restrict__ const b)
{
    //HINT: Look at copyKernel above

    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col

    int index_in  =  i*cols+j;      // Compute input index (i,j) from matrix A
    int index_out =  j*rows+i;      // Compute output index (j,i) in matrix B = transpose(A)

    // Copy data from A to B
    b[index_out] = a[index_in];
}

__global__ void matrixTransposeShared(const float* __restrict__ const a,
        float* __restrict__ const b)
{
    //Allocate appropriate shared memory
    __shared__ float mat[BLOCK_SIZE_X][BLOCK_SIZE_Y];

    //Compute input and output index
    int bx = blockIdx.x * BLOCK_SIZE_X;
    int by = blockIdx.y * BLOCK_SIZE_Y;
    int i  = by + threadIdx.y;        // row
    int j  = bx + threadIdx.x;        // col
    int ti = bx + threadIdx.y;        // row
    int tj = by + threadIdx.x;        // col

    //Copy data from input to shared memory
    if(i < rows && j < cols)
            mat[threadIdx.x][threadIdx.y] = a[i * cols + j];

    __syncthreads();

    //Copy data from shared memory to global memory
    if(tj < cols && ti < rows)
            b[ti * rows + tj] = mat[threadIdx.y][threadIdx.x];
}

__global__ void matrixTransposeSharedwBC(const float* __restrict__ const a,
        float* __restrict__ const b)
{
    //HINT: Copy code from matrixTransposeShared kernel, while solving bank conflict problem
    __shared__ float mat[BLOCK_SIZE_X][BLOCK_SIZE_Y + 1];

    //Allocate appropriate shared memory

    //Compute input and output index
    int bx = blockIdx.x * BLOCK_SIZE_X;
    int by = blockIdx.y * BLOCK_SIZE_Y;
    int i  = by + threadIdx.y;        // row
    int j  = bx + threadIdx.x;        // col
    int ti = bx + threadIdx.y;        // row
    int tj = by + threadIdx.x;        // col

    //Copy data from input to shared memory
    if(i < rows && j < cols)
            mat[threadIdx.x][threadIdx.y] = a[i * cols + j];

    __syncthreads();

    //Copy data from shared memory to global memory
    if(tj < cols && ti < rows)
            b[ti * rows + tj] = mat[threadIdx.y][threadIdx.x];
}

__global__ void matrixTransposeUnrolled(const float* __restrict__ const a,
        float* __restrict__ const b)
{
    //Allocate appropriate shared memory
    __shared__ float mat[TILE][TILE + 1];

    //Compute input and output index
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    //Copy data from input to shared memory. Multiple copies per thread.
    #pragma unroll
    for(int k = 0; k < TILE ; k += SIDE)
    {
            if(x < rows && y + k < cols)
                    mat[threadIdx.y + k][threadIdx.x] = a[((y + k) * rows) + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    //Copy data from shared memory to global memory. Multiple copies per thread.
    #pragma unroll
    for(int k = 0; k < TILE; k += SIDE)
    {
            if(x < cols && y + k < rows)
                    b[(y + k) * cols + x] = mat[threadIdx.x][threadIdx.y + k];
    }
}

int main(int argc, char *argv[])
{
    //Run Memcpy benchmarks
    nvtxRangeId_t cudaBenchmark = nvtxRangeStart("CUDA Memcpy Benchmark");
#if defined WIN64
    system(".\\..\\bin\\cudaBenchmark.exe");
#elif defined LINUX
    system("./bin/cudaBenchmark");
#endif
    nvtxRangeEnd(cudaBenchmark);

    // Host arrays.
    float* a = new float[rows*cols];
    float* b = new float[rows*cols];
    float* a_gold = new float[rows*cols];
    float* b_gold = new float[rows*cols];

    // Device arrays
    float *d_a, *d_b;

    // Allocate memory on the device
    CUDA( cudaMalloc((void **) &d_a, rows*cols*sizeof(float)) );

    CUDA( cudaMalloc((void **) &d_b, rows*cols*sizeof(float)) );

    // Fill matrix A
    for (int i = 0; i < rows * cols; i++)
        a[i] = (float)i;

    cout << endl;

    // Copy array contents of A from the host (CPU) to the device (GPU)
    cudaMemcpy(d_a, a, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    //Compute "gold" reference standard
    for(int ii = 0; ii < rows; ii++)
    {
        for(int jj = 0; jj < cols; jj++)
        {
            a_gold[jj * rows + ii] = a[jj * cols + ii];
            b_gold[ii * cols + jj] = a[jj * cols + ii];
        }
    }

    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cout << "***Launch the transpose!***" << endl << endl;

#define CPU_TRANSPOSE
#ifdef CPU_TRANSPOSE
    ////////////////////////////////////////////////////////////
    cout << "******************************************" << endl;
    cout << "***CPU Transpose***" << endl;
    {
        // start the timer
        nvtxRangeId_t cpuBenchmark = nvtxRangeStart("CPU Transpose Benchmark");

        clock_t begin = clock();
        int iters = 10;
        for (int k=0; k<iters; k++)
        {
            for(int ii = 0; ii < rows; ii++)
                for(int jj = 0; jj < cols; jj++)
                    b[ii * cols + jj] = a[jj * cols + ii];
        }
        // stop the timer
        clock_t end = clock();
        nvtxRangeEnd(cpuBenchmark);

        float time = 0.0f;
        time = diffclock(end, begin);

        // print out the time required for the kernel to finish the transpose operation
        double Bandwidth = (double)iters*2.0*1000.0*(double)(rows*cols*sizeof(float)) / (1000.0*1000.0*1000.0*time);
        cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
        cout << "Bandwidth (GB/s) = " << Bandwidth << endl;
    }
    cout << "******************************************" << endl;
    cout << endl;
    ////////////////////////////////////////////////////////////
#endif

    ////////////////////////////////////////////////////////////
    cout << "******************************************" << endl;
    cout << "***Device To Device Copy***" << endl;
    {
        preprocess(b, d_b, rows*cols);
        // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
        // Calculate number of blocks along X and Y in a 2D CUDA "grid"

        DIMS dims;
        dims.dimBlock = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
        dims.dimGrid  = dim3(divup(rows, BLOCK_SIZE_X),
                             divup(cols, BLOCK_SIZE_Y),
                             1
                             );

        // start the timer
        nvtxRangeId_t naiveBenchmark = nvtxRangeStart("Device to Device Copy");
        cudaEventRecord( start, 0);

        int iters = 10;
        for (int i=0; i<iters; i++)
        {
            // Launch the GPU kernel
            copyKernel<<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);
        }
        // stop the timer
        cudaEventRecord( stop, 0);
        cudaEventSynchronize( stop );
        nvtxRangeEnd(naiveBenchmark);

        float time = 0.0f;
        cudaEventElapsedTime( &time, start, stop);

        // print out the time required for the kernel to finish the transpose operation
        double Bandwidth = (double)iters*2.0*1000.0*(double)(rows*cols*sizeof(float)) /
                            (1000.0*1000.0*1000.0*time);        //2.0 for read of A and read and write of B
        cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
        cout << "Bandwidth (GB/s) = " << Bandwidth << endl;

        // copy the answer back to the host (CPU) from the device (GPU)
        cudaMemcpy(b, d_b, cols*rows*sizeof(float), cudaMemcpyDeviceToHost);

        postprocess(a_gold, b, rows * cols);
    }
    cout << "******************************************" << endl;
    cout << endl;
    ////////////////////////////////////////////////////////////

#if NAIVE_TRANSPOSE
    ////////////////////////////////////////////////////////////
    cout << "******************************************" << endl;
    cout << "***Naive Transpose***" << endl;
    {
        preprocess(b, d_b, rows*cols);
        // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
        // Calculate number of blocks along X and Y in a 2D CUDA "grid"
        // HINT: Look above for copy kernel dims computation
        DIMS dims;
        dims.dimBlock = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
        dims.dimGrid  = dim3(divup(rows, BLOCK_SIZE_X),
                             divup(cols, BLOCK_SIZE_Y),
                             1);

        // start the timer
        nvtxRangeId_t naiveBenchmark = nvtxRangeStart("Naive Transpose Benchmark");
        cudaEventRecord( start, 0);

        int iters = 10;
        for (int i=0; i<iters; i++)
        {
            // Launch the GPU kernel
            matrixTransposeNaive<<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);
        }
        // stop the timer
        cudaEventRecord( stop, 0);
        cudaEventSynchronize( stop );
        nvtxRangeEnd(naiveBenchmark);

        float time = 0.0f;
        cudaEventElapsedTime( &time, start, stop);

        // print out the time required for the kernel to finish the transpose operation
        double Bandwidth = (double)iters*2.0*1000.0*(double)(rows*cols*sizeof(float)) /
            (1000.0*1000.0*1000.0*time);
        cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
        cout << "Bandwidth (GB/s) = " << Bandwidth << endl;

        // copy the answer back to the host (CPU) from the device (GPU)
        cudaMemcpy(b, d_b, cols*rows*sizeof(float), cudaMemcpyDeviceToHost);

        postprocess(b_gold, b, rows * cols);
    }
    cout << "******************************************" << endl;
    cout << endl;
    ////////////////////////////////////////////////////////////
#endif

#if SHARED_MEM_TRANSPOSE
    ////////////////////////////////////////////////////////////
    cout << "******************************************" << endl;
    cout << "***Shared Memory Transpose***" << endl;
    {
        preprocess(b, d_b, rows*cols);
        // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
        // Calculate number of blocks along X and Y in a 2D CUDA "grid"
        DIMS dims;
        dims.dimBlock = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
        dims.dimGrid  = dim3(divup(rows, BLOCK_SIZE_X),
                             divup(cols, BLOCK_SIZE_Y),
                             1);

        // start the timer
        nvtxRangeId_t sharedMemBenchmark = nvtxRangeStart("Shared Memory Transpose Benchmark");
        cudaEventRecord( start, 0);

        int iters = 10;
        for (int i=0; i<iters; i++)
        {
            // Launch the GPU kernel
            matrixTransposeShared<<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);
        }
        // stop the timer
        cudaEventRecord( stop, 0);
        cudaEventSynchronize( stop );
        nvtxRangeEnd(sharedMemBenchmark);

        float time = 0.0f;
        cudaEventElapsedTime( &time, start, stop);

        // print out the time required for the kernel to finish the transpose operation
        double Bandwidth = (double)iters*2.0*1000.0*(double)(rows*cols*sizeof(float)) /
            (1000.0*1000.0*1000.0*time);
        cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
        cout << "Bandwidth (GB/s) = " << Bandwidth << endl;

        // copy the answer back to the host (CPU) from the device (GPU)
        cudaMemcpy(b, d_b, cols*rows*sizeof(float), cudaMemcpyDeviceToHost);

        postprocess(b_gold, b, rows * cols);
    }
    cout << "******************************************" << endl;
    cout << endl;
    ////////////////////////////////////////////////////////////
#endif

#if BANK_CONF_TRANSPOSE
    ////////////////////////////////////////////////////////////
    cout << "******************************************" << endl;
    cout << "***Without Bank Conflicts Transpose***" << endl;
    {
        preprocess(b, d_b, rows*cols);
        // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
        // Calculate number of blocks along X and Y in a 2D CUDA "grid"
        DIMS dims;
        dims.dimBlock = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
        dims.dimGrid  = dim3(divup(rows, BLOCK_SIZE_X),
                             divup(cols, BLOCK_SIZE_Y),
                             1);

        // start the timer
        nvtxRangeId_t sharedMemBenchmark = nvtxRangeStart("Shared Memory Transpose Benchmark");
        cudaEventRecord( start, 0);

        int iters = 10;
        for (int i=0; i<iters; i++)
        {
            // Launch the GPU kernel
            matrixTransposeSharedwBC<<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);
        }
        // stop the timer
        cudaEventRecord( stop, 0);
        cudaEventSynchronize( stop );
        nvtxRangeEnd(sharedMemBenchmark);

        float time = 0.0f;
        cudaEventElapsedTime( &time, start, stop);

        // print out the time required for the kernel to finish the transpose operation
        double Bandwidth = (double)iters*2.0*1000.0*(double)(rows*cols*sizeof(float)) /
            (1000.0*1000.0*1000.0*time);
        cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
        cout << "Bandwidth (GB/s) = " << Bandwidth << endl;

        // copy the answer back to the host (CPU) from the device (GPU)
        cudaMemcpy(b, d_b, cols*rows*sizeof(float), cudaMemcpyDeviceToHost);

        postprocess(b_gold, b, rows * cols);
    }
    cout << "******************************************" << endl;
    cout << endl;
    ////////////////////////////////////////////////////////////
#endif

#if UNROLLED_TRANSPOSE
    ////////////////////////////////////////////////////////////
    cout << "******************************************" << endl;
    cout << "***Unrolled Loops Transpose***" << endl;
    {
        preprocess(b, d_b, rows*cols);
        // Assign a 2D distribution of TILE x SIDE x 1 CUDA threads within
        // Calculate number of blocks along X and Y in a 2D CUDA "grid"
        DIMS dims;
        dims.dimBlock = dim3(TILE, SIDE, 1);
        dims.dimGrid  = dim3(divup(rows, TILE),
                             divup(cols, TILE),
                             1);

        // start the timer
        nvtxRangeId_t unrolledBenchmark = nvtxRangeStart("Shared Memory Transpose Benchmark");
        cudaEventRecord( start, 0);

        int iters = 10;
        for (int i=0; i<iters; i++)
        {
            // Launch the GPU kernel
            matrixTransposeUnrolled<<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);
        }
        // stop the timer
        cudaEventRecord( stop, 0);
        cudaEventSynchronize( stop );
        nvtxRangeEnd(unrolledBenchmark);

        float time = 0.0f;
        cudaEventElapsedTime( &time, start, stop);

        // print out the time required for the kernel to finish the transpose operation
        double Bandwidth = (double)iters*2.0*1000.0*(double)(rows*cols*sizeof(float)) /
            (1000.0*1000.0*1000.0*time);
        cout << "Elapsed Time for " << iters << " runs = " << time << "ms" << endl;
        cout << "Bandwidth (GB/s) = " << Bandwidth << endl;

        // copy the answer back to the host (CPU) from the device (GPU)
        cudaMemcpy(b, d_b, cols*rows*sizeof(float), cudaMemcpyDeviceToHost);

        postprocess(b_gold, b, rows * cols);
    }
    cout << "******************************************" << endl;
    cout << endl;
    ////////////////////////////////////////////////////////////
#endif
    // copy the answer back to the host (CPU) from the device (GPU)

    /*
       cout << "Entries of B: \n";
       for (int i = 0; i < 32; i++) {
       cout << b[i] << " ";
       }
       cout << endl;
       for (int i = 0; i < 32; i++) {
       cout << b[i * cols] << " ";
       }
       cout << endl;

     */

    // free device memory
    cudaFree(d_a);
    cudaFree(d_b);

    // free host memory
    delete[] a;
    delete[] b;

    //Destroy Events
    CUDA(cudaEventDestroy(start));
    CUDA(cudaEventDestroy(stop));

    //CUDA Reset for NVProf
    CUDA(cudaDeviceReset());

    // successful program termination
    return 0;
}
