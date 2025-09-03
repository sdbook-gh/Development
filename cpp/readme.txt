## 不使用调试器打印虚函数调用（GCC、Clang）
# 打印调用函数的地址
  static void print_caller_addr() {
    void* addr = __builtin_return_address(0);
    printf("caller addr: %p\n", addr);
  }
# 获取虚函数地址
  printf("Dog::makeSound %p\n", (void*)&Dog::makeSound);
# 获取虚函数表地址
  void* vptr = *(void**)pAnimal; // *(void**)&Animal
# 打印虚函数表表项
  if (std::is_polymorphic<IAnimal>::value) {
    void* vptr = *(void**)pAnimal;
    printf("vptr: %p\n", vptr);
    printf("vptr[0]: %p\n", ((void **)vptr)[0]);
    printf("vptr[1]: %p\n", ((void **)vptr)[1]);
  }
