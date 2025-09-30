#include <iostream>
#include <cstdio>
#include <typeinfo>
#include <dlfcn.h>

static void print_caller_addr() {
  void* addr = __builtin_return_address(0);
  printf("caller addr: %p\n", addr);
}

extern "C" void test_func() { std::cout << "test_func" << std::endl; }
extern "C" void test_func_new() { std::cout << "test_func_new" << std::endl; }

// 定义接口类
class IAnimal {
public:
  // 纯虚函数声明
  virtual void makeSound() = 0;
  virtual void move() = 0;
  // 虚析构函数
  virtual ~IAnimal() {}
};

// 实现类1：狗
class Dog : public IAnimal {
public:
  void makeSound() override;

  void move() override;
};

void Dog::makeSound() {
  print_caller_addr();
  std::cout << "汪汪!" << std::endl;
}
void Dog::move() { std::cout << "狗在跑步" << std::endl; }

// 实现类2：猫
class Cat : public IAnimal {
public:
  void makeSound() override;

  void move() override;
};

void Cat::makeSound() {
  print_caller_addr();
  std::cout << "喵喵!" << std::endl;
}
void Cat::move() { std::cout << "猫在走路" << std::endl; }

int main() {
  printf("test_func %p\n", &test_func);
  printf("test_func_new %p\n", &test_func_new);
  {
    printf("Dog::makeSound %p\n", (void*)&Dog::makeSound);
    printf("Cat::makeSound %p\n", (void*)&Cat::makeSound);
    IAnimal* pAnimal = new Dog();
    printf("pAnimal: %s\n", typeid(*pAnimal).name());
    if (std::is_polymorphic<IAnimal>::value) {
      void* vptr = *(void**)pAnimal;
      printf("vptr: %p\n", vptr);
      printf("vptr[0]: %p\n", ((void **)vptr)[0]);
      printf("vptr[1]: %p\n", ((void **)vptr)[1]);
    }
    Dl_info info;
    void* makeSound_addr = (void*)&Dog::makeSound;
    if (dladdr(makeSound_addr, &info)) {
      // 计算函数地址相对于库基地址的偏移量
      size_t offset = (size_t)makeSound_addr - (size_t)info.dli_fbase;
      // 打印函数所在的共享库文件名、库的基地址和函数偏移量
      printf("库文件: %s\n", info.dli_fname);
      printf("基地址: %p\n", info.dli_fbase);
      printf("函数偏移: %zx\n", offset);
      printf("makeSound_addr: %p\n", makeSound_addr);
    }
    // 调用函数并清理
    pAnimal->makeSound();
    delete pAnimal;
  }
  return 0;
}
