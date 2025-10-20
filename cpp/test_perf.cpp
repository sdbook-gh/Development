#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <chrono>
#include <thread>
#include <string>
#include <span>
#include <ranges>
#include <memory>
#include <immintrin.h>

inline constexpr std::string TEST_MODE = "AVX2_COPY";
__attribute__((noinline)) void efficientCopy(const uint8_t* src, uint8_t* dst, size_t srcSize) {
  if constexpr (TEST_MODE == "SKIP_COPY") {
    static bool _ret = []() {
      printf("SKIP_COPY\n");
      return true;
    }();
    const size_t COPY_SIZE = 1000; // 每次复制1000字节
    const size_t SKIP_SIZE = 365; // 每次跳过365字节
    const size_t STRIDE = COPY_SIZE + SKIP_SIZE; // 步长1365字节
    size_t srcPos = 0;
    size_t dstPos = 0;
    // 主循环：批量memcpy
    // #pragma unroll
    while (srcPos + COPY_SIZE <= srcSize) {
      memcpy(dst + dstPos, src + srcPos, COPY_SIZE);
      srcPos += STRIDE;
      dstPos += COPY_SIZE;
    }
    // 处理剩余部分（如果有）
    size_t remaining = srcSize - srcPos;
    if (remaining > 0) { memcpy(dst + dstPos, src + srcPos, remaining); }
  } else if constexpr (TEST_MODE == "DIRECT_COPY") {
    static bool _ret = []() {
      printf("DIRECT_COPY\n");
      return true;
    }();
    memcpy(dst, src, srcSize);
  } else if constexpr (TEST_MODE == "AVX2_COPY") {
    static bool _ret = []() {
      printf("AVX2_COPY\n");
      return true;
    }();
    const size_t COPY_SIZE = 1000;
    const size_t SKIP_SIZE = 365;
    const size_t STRIDE = COPY_SIZE + SKIP_SIZE;
    size_t srcPos = 0;
    size_t dstPos = 0;
    while (srcPos + COPY_SIZE <= srcSize) {
      const uint8_t* srcBlock = src + srcPos;
      uint8_t* dstBlock = dst + dstPos;
      size_t i = 0;
      // AVX2：每次复制32字节
      for (; i + 32 <= COPY_SIZE; i += 32) {
        __m256i data = _mm256_loadu_si256((__m256i*)(srcBlock + i));
        _mm256_storeu_si256((__m256i*)(dstBlock + i), data);
      }
      // 处理剩余字节（1000 % 32 = 8字节）
      for (; i < COPY_SIZE; i++) { dstBlock[i] = srcBlock[i]; }
      srcPos += STRIDE;
      dstPos += COPY_SIZE;
    }
    // 处理最后不完整的块
    size_t remaining = srcSize - srcPos;
    if (remaining > 0) { memcpy(dst + dstPos, src + srcPos, remaining); }
  } else if constexpr (TEST_MODE == "SSE2_COPY") {
    static bool _ret = []() {
      printf("SSE2_COPY\n");
      return true;
    }();
    const size_t COPY_SIZE = 1000;
    const size_t SKIP_SIZE = 365;
    const size_t STRIDE = COPY_SIZE + SKIP_SIZE;
    size_t srcPos = 0;
    size_t dstPos = 0;
    while (srcPos + COPY_SIZE <= srcSize) {
      const uint8_t* srcBlock = src + srcPos;
      uint8_t* dstBlock = dst + dstPos;
      size_t i = 0;
      // SSE2：每次复制16字节
      for (; i + 16 <= COPY_SIZE; i += 16) {
        __m128i data = _mm_loadu_si128((__m128i*)(srcBlock + i));
        _mm_storeu_si128((__m128i*)(dstBlock + i), data);
      }
      // 处理剩余字节（1000 % 16 = 8字节）
      for (; i < COPY_SIZE; i++) { dstBlock[i] = srcBlock[i]; }
      srcPos += STRIDE;
      dstPos += COPY_SIZE;
    }
    // 处理最后不完整的块
    size_t remaining = srcSize - srcPos;
    if (remaining > 0) { memcpy(dst + dstPos, src + srcPos, remaining); }
  } else if constexpr (TEST_MODE == "AVX512_COPY") {
    static bool _ret = []() {
      printf("AVX512_COPY\n");
      return true;
    }();
    const size_t COPY_SIZE = 1000;
    const size_t SKIP_SIZE = 365;
    const size_t STRIDE = COPY_SIZE + SKIP_SIZE;
    size_t srcPos = 0;
    size_t dstPos = 0;
    while (srcPos + COPY_SIZE <= srcSize) {
      const uint8_t* srcBlock = src + srcPos;
      uint8_t* dstBlock = dst + dstPos;
      size_t i = 0;
      // AVX-512：每次复制64字节
      for (; i + 64 <= COPY_SIZE; i += 64) {
        __m512i data = _mm512_loadu_si512(srcBlock + i);
        _mm512_storeu_si512(dstBlock + i, data);
      }
      // 处理剩余字节
      for (; i < COPY_SIZE; i++) { dstBlock[i] = srcBlock[i]; }
      srcPos += STRIDE;
      dstPos += COPY_SIZE;
    }
    size_t remaining = srcSize - srcPos;
    if (remaining > 0) { memcpy(dst + dstPos, src + srcPos, remaining); }
  }
}

__attribute__((noinline)) void copy_skip_modern(std::span<const uint8_t> src, std::span<uint8_t> dst) {
  static bool _ret = []() {
    printf("MODERN_COPY\n");
    return true;
  }();
  constexpr size_t block_size = 1000;
  constexpr size_t copy_size = 635;
  // 计算目标数组大小
  size_t num_blocks = src.size() / block_size;
  size_t dst_offset = 0;
  for (auto block : src | std::views::chunk(block_size)) {
    if (block.size() < copy_size) break; // 防止最后不足一块
    std::memcpy(dst.data() + dst_offset, block.data(), copy_size);
    dst_offset += copy_size;
  }
}

int main() {
  std::vector<uint8_t> src;
  std::vector<uint8_t> dst;
  src.resize(1000064);
  dst.resize(1000064);
  void* srcVec = src.data();
  auto srcVecSize = src.size();
  std::align(32, sizeof(uint8_t), srcVec, srcVecSize);
  void* dstVec = dst.data();
  auto dstVecSize = dst.size();
  std::align(32, sizeof(uint8_t), dstVec, dstVecSize);
  printf("srcVecSize: %d, dstVecSize: %d\n", srcVecSize, dstVecSize);
  alignas(32) uint8_t* srcArray = new (std::align_val_t(32)) uint8_t[1000000];
  alignas(32) uint8_t* dstArray = new (std::align_val_t(32)) uint8_t[1000000];
  // uint8_t* srcArray = new uint8_t[1000000];
  // uint8_t* dstArray = new uint8_t[1000000];
  memset(srcArray, 0, 1000000);
  memset(dstArray, 0, 1000000);
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
      efficientCopy(srcArray, dstArray, 1000000);
      // copy_skip_modern({srcArray, 1000000}, {dstArray, 1000000});
    }
    auto finish = std::chrono::high_resolution_clock::now();
    int seconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    printf("elapsed: %d ms\n", seconds);
  }
  {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
      // efficientCopy((uint8_t*)srcVec, (uint8_t*)dstVec, 1000000);
      copy_skip_modern({(uint8_t*)srcVec, 1000000}, {(uint8_t*)dstVec, 1000000});
    }
    auto finish = std::chrono::high_resolution_clock::now();
    int seconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    printf("elapsed: %d ms\n", seconds);
  }
  delete[] srcArray;
  delete[] dstArray;
  return 0;
}
