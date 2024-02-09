#pragma once
#include <cstdint>

extern "C" int addWithCuda(int *c, const int *a, const int *b, unsigned int size);
extern "C" int resetCudaDevice();
extern "C" int processWithCuda(uint8_t *src_data, uint32_t rows, uint32_t cols, uint8_t *dst_data);
extern "C" void grayWithCuda(uint8_t *src_data, uint32_t rows, uint32_t cols, uint8_t *dst_data);
