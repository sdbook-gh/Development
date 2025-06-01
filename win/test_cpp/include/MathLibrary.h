#pragma once

#ifdef MATHLIBRARY_EXPORTS
#define MATHLIBRARY_API __declspec(dllexport)
#else
#define MATHLIBRARY_API __declspec(dllimport)
#endif

extern "C" MATHLIBRARY_API int add(int a, int b);        // 显式导出函数
extern "C" MATHLIBRARY_API int subtract(int a, int b);

// 类导出示例（可选）
class MATHLIBRARY_API Calculator {
public:
    int multiply(int a, int b);
};
