#include <stdio.h>

#include "api.h"

__global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
extern "C" int addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

extern "C" int resetCudaDevice() {
    cudaError_t cudaStatus;
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}

// __global__ void Traverse(uchar3 *_src_dev, uchar3 *dst_dev, int col, int row) {
//     // 一维数据索引计算（万能计算方法）
//     int tid = blockIdx.z * (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z) +
//               blockIdx.y * gridDim.x * (blockDim.x * blockDim.y * blockDim.z) +
//               blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) +
//               threadIdx.y * blockDim.x + threadIdx.x;
//     if (tid < col * row * 3) {
//         dst_dev[tid].x = 0.3 * _src_dev[tid].x;
//         dst_dev[tid].y = 0.6 * _src_dev[tid].y;
//         dst_dev[tid].z = 0.1 * _src_dev[tid].z;
//     }
// }

// __global__ void Traverse(uchar3 *_src_dev, uchar3 *dst_dev, int col, int row) {
//     // 一维数据索引计算（万能计算方法）
//     int tid = blockIdx.z * (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z) +
//               blockIdx.y * gridDim.x * (blockDim.x * blockDim.y * blockDim.z) +
//               blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) +
//               threadIdx.y * blockDim.x + threadIdx.x;
//     if (tid < col * row * 3) {
//         dst_dev[tid].x = 255 - _src_dev[tid].x;
//         dst_dev[tid].y = 255 - _src_dev[tid].y;
//         dst_dev[tid].z = 255 - _src_dev[tid].z;
//     }
// }

__global__ void Traverse(uchar3 *_src_dev, uchar3 *dst_dev, int col, int row) {
    // 一维数据索引计算（万能计算方法）
    int tid = blockIdx.z * (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z) +
              blockIdx.y * gridDim.x * (blockDim.x * blockDim.y * blockDim.z) +
              blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) + threadIdx.z * (blockDim.x * blockDim.y) +
              threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < col * row * 3) {
        dst_dev[tid].x = 0.3 * _src_dev[tid].x;
        dst_dev[tid].y = 0.6 * _src_dev[tid].y;
        dst_dev[tid].z = 0.1 * _src_dev[tid].z;
    }
}

extern "C" int processWithCuda(uint8_t *src_data, uint32_t rows, uint32_t cols, uint8_t *dst_data) {
    uchar3 *src_dev, *dst_dev;
    cudaMalloc((void **)&src_dev, rows * cols * sizeof(uchar3));
    cudaMalloc((void **)&dst_dev, rows * cols * sizeof(uchar3));
    cudaMemcpy(src_dev, src_data, rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice);
    dim3 grid(1 + (cols * rows / (32 * 32 + 1)), 1, 1); // grid
    dim3 block(32, 32, 1);

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);
    Traverse<<<grid, block>>>(src_dev, dst_dev, cols, rows);
    cudaEventSynchronize(stop1);
    cudaEventRecord(stop1, 0);
    float time1;
    cudaEventElapsedTime(&time1, start1, stop1);
    printf("Gpu所耗费的时间: %fms\n", time1);
    cudaMemcpy(dst_data, dst_dev, rows * cols * sizeof(uchar3), cudaMemcpyDeviceToHost);

    // free
    cudaFree(src_dev);
    cudaFree(dst_dev);
    return 0;
}

__global__ void rgb2grayincuda(uchar3 *const d_in, uint8_t *const d_out, uint rows, uint cols) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        uchar3 rgb = d_in[idy * cols + idx];
        d_out[idy * cols + idx] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
}

extern "C" void grayWithCuda(uint8_t *src_data, uint32_t rows, uint32_t cols, uint8_t *dst_data) {
    uchar3 *d_in;
    unsigned char *d_out;

    cudaMalloc((void **)&d_in, rows * cols * sizeof(uchar3));
    cudaMalloc((void **)&d_out, rows * cols * sizeof(uint8_t));

    cudaMemcpy(d_in, src_data, rows * cols * sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32, 1);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    clock_t start, end;
    start = clock();

    rgb2grayincuda<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, rows, cols);

    cudaDeviceSynchronize();
    end = clock();

    printf("cuda exec time is %.8f\n", (double)(end - start) / CLOCKS_PER_SEC);

    cudaMemcpy(dst_data, d_out, rows * cols * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}
