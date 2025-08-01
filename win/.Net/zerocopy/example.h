#pragma once
#include <cstdint>
#include <iostream>

extern "C" __declspec(dllexport) void ProcessDataZeroCopy(uint8_t* data, int size);
extern "C" __declspec(dllexport) uint8_t* CreateArray(int* outLength);
extern "C" __declspec(dllexport) void DestroyArray(uint8_t* arr);
