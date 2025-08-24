#include <iostream>
#include <vector>
#include <cstdlib>
// 错误检查宏
// 在每次 CUDA API 调用后检查是否有错误发生
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#define CUDA_CHECK(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
// CUDA Kernel: 并行执行向量加法
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
  // 计算全局唯一的线程 ID
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // 确保线程 ID 不会越过数组边界
  // 因为我们启动的线程总数可能略多于实际需要的元素数
  if (i < N) { C[i] = A[i] + B[i]; }
}
int main() {
  const int N = 1 << 16; // 向量的大小，例如 2^16 = 65536
  const size_t vectorBytes = N * sizeof(float);
  // 1. 在主机 (CPU) 上分配内存
  std::vector<float> h_A(N);
  std::vector<float> h_B(N);
  std::vector<float> h_C(N);
  // 2. 初始化主机上的向量
  for (int i = 0; i < N; ++i) {
    h_A[i] = rand() % 100;
    h_B[i] = rand() % 100;
  }
  // 3. 在设备 (GPU) 上分配内存
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, vectorBytes));
  CUDA_CHECK(cudaMalloc(&d_B, vectorBytes));
  CUDA_CHECK(cudaMalloc(&d_C, vectorBytes));
  // 4. 将数据从主机内存拷贝到设备内存
  // cudaMemcpyHostToDevice: 从主机到设备
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), vectorBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), vectorBytes, cudaMemcpyHostToDevice));
  // 5. 设置线程配置并启动 Kernel
  int threadsPerBlock = 256;
  // 计算需要的块数，确保能覆盖所有 N 个元素
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "Launching Kernel with " << blocksPerGrid << " blocks and " << threadsPerBlock << " threads per block." << std::endl;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
  // 等待 Kernel 执行完毕
  CUDA_CHECK(cudaGetLastError()); // 检查内核启动是否有错误
  CUDA_CHECK(cudaDeviceSynchronize()); // 等待内核执行完成
  // 6. 将计算结果从设备内存拷贝回主机内存
  // cudaMemcpyDeviceToHost: 从设备到主机
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, vectorBytes, cudaMemcpyDeviceToHost));
  // 7. 验证结果
  bool correct = true;
  for (int i = 0; i < 10; ++i) { // 只检查前 10 个元素作为示例
    if (abs((h_A[i] + h_B[i]) - h_C[i]) > 1e-6) {
      std::cerr << "Verification failed at index " << i << "! " << h_A[i] << " + " << h_B[i] << " = " << h_A[i] + h_B[i] << " but got " << h_C[i] << std::endl;
      correct = false;
      break;
    }
  }
  if (correct) {
    std::cout << "Verification successful!" << std::endl;
    std::cout << "Example results:" << std::endl;
    for (int i = 0; i < 5; ++i) { std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl; }
  }
  // 8. 释放设备上的内存
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  return 0;
}
