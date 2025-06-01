#include "MathLibrary.h"
#include <iostream>

int main() {
    std::cout << "5 + 3 = " << add(5, 3) << std::endl;  // 调用 DLL 函数
    
    Calculator calc;
    std::cout << "5 * 3 = " << calc.multiply(5, 3) << std::endl;
    return 0;
}
