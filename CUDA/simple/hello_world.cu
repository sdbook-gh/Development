#include <cstdio>

// CUDA Kernel: 在 GPU 上执行的函数
// __global__ 声明符表示这个函数将从 CPU 调用，但在 GPU 上执行
__global__ void hello_from_gpu() {
  // blockIdx.x 提供了当前线程所在块的索引
  // blockDim.x 提供了每个块中的线程数
  // threadIdx.x 提供了当前线程在块内的索引
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Hello World from thread %d blockIdx.x %d blockDim.x %d threadIdx.x %d\n", tid, blockIdx.x, blockDim.x, threadIdx.x);
}
int main() {
  // 在主函数 (CPU 端) 中
  printf("girdDim: %d, blockDim: %d\n", 2, 4);
  // <<<...>>> 是 CUDA 的执行配置语法，用于启动 kernel
  // 第一个参数: 网格中的块数 (Number of blocks in the grid)
  // 第二个参数: 每个块中的线程数 (Number of threads per block)
  // 在这里，我们启动了 2 个块，每个块有 4 个线程，总共 8 个 GPU 线程。
  hello_from_gpu<<<2, 4>>>();
  // cudaDeviceSynchronize() 会让 CPU 等待所有在 GPU 上启动的异步任务完成
  // 如果不加这个，CPU 端的 main 函数可能会在 GPU 完成打印前就结束了
  cudaDeviceSynchronize();
  return 0;
}
