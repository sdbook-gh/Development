https://www.kimi.com/chat/d2kq9tbduqbctmlecofg

export PATH=/usr/local/cuda-12.8/bin:$PATH

nvcc hello_world.cu -o hello_world

# CUDA 编程例子及解释
https://gemini.google.com/app/d7a38fcc388092d8 # very well
# choose grid and block dimensions
  在 CUDA 中计算所需块数 (blocksPerGrid) 的原理和通用方法。
    核心原理：确保完全覆盖
      CUDA 编程的核心思想是将数据并行化。对于一个有 N 个元素的数据集（比如一个大小为 N 的数组），我们的目标是启动至少 N 个线程，让每个线程负责处理一个元素。
      threadsPerBlock (每个块的线程数): 这是我们自己决定的一个参数。通常选择 32 的倍数（如 128, 256, 512），因为 GPU 的线程是以 32 个为一组（称为 "warp"）来执行的。这个值是固定的。
      blocksPerGrid (网格中的块数): 这是我们需要计算的参数。我们的目标是计算出最少需要多少个块，才能保证总线程数大于或等于 N。
        总线程数由以下公式得出：
        Total Threads = blocksPerGrid * threadsPerBlock
        为了处理所有 N 个元素，我们必须满足：
        Total Threads ≥ N
        因此，我们需要的块数 blocksPerGrid 必须满足：
        blocksPerGrid ≥ threadsPerBlock / N，由于块数必须是整数，所以 blocksPerGrid 必须是 这个除法结果的向上取整 (Ceiling)。
          ​int threadsPerBlock = 256;
          // 计算需要的块数，确保能覆盖所有 N 个元素
          int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // 达到向上取整的目的
# 核函数为什么要有多余的线程和边界检查
  在 N = 1025, threadsPerBlock = 256 的例子中：
  我们启动了 5 * 256 = 1280 个线程。
  但我们只有 1025 个元素需要处理。
  这意味着线程 ID 从 0 到 1024 的线程是有任务的，而线程 ID 从 1025 到 1279 的线程是多余的，它们不应该去访问数组。
  这就是为什么在 CUDA 内核函数 (kernel) 中，必须要有一个边界检查：
    __global__ void myKernel(float* data, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        // 边界检查：确保当前线程ID没有超出有效数据范围
        if (i < N) {
            // 只有在数据范围内的线程才执行操作
            data[i] = ...;
        }
    }
  这个 if (i < N) 的判断至关重要。它保证了只有那些被分配了有效任务的线程才会执行计算和内存访问，从而防止了多余线程访问数组越界，避免程序崩溃或产生错误数据。
