#include "example.h"

extern "C" __declspec(dllexport) void ProcessDataZeroCopy(uint8_t* data, int size) {
  for (int i = 0; i < size; ++i) { std::cout << "Byte " << i << ": " << data[i] << std::endl; }
}

extern "C" __declspec(dllexport) uint8_t* CreateArray(int* outLength) {
  int len = 5;
  *outLength = len;
  uint8_t* arr = new uint8_t[*outLength];
  for (int i = 0; i < len; ++i) { arr[i] = i * 2; }
  return arr;
}

extern "C" __declspec(dllexport) void DestroyArray(uint8_t* arr) { delete[] arr; }
