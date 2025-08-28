#include <initializer_list>
#include <iostream>

template <typename T, typename... Args>
void process_variadic_arg1(Args&&... args) {
  static_assert((std::is_convertible_v<Args, T> && ...), "All arguments must be convertible to T");
  std::initializer_list<T>{(std::cout << args << std::endl, T(std::forward<Args>(args)))...};
}

template <typename... Args>
void process_variadic_arg2(Args&&... args) {
  std::initializer_list<int>{[&] {
    std::cout << args << std::endl;
    return 0;
  }()...};
}

#include <type_traits>
#include <string>

// #define CHECK_VIRTUAL_STRUCTURE
#ifdef CHECK_VIRTUAL_STRUCTURE
struct Base {
  virtual void do_something() = 0;
  virtual ~Base() = default;
};
#endif

struct A
#ifdef CHECK_VIRTUAL_STRUCTURE
  : public Base
#endif
{
  int a;
  int b;
  A(const A& other) : a(other.a), b(other.b) {}
  A& operator=(const A& other) {
    if (this != &other) {
      a = other.a;
      b = other.b;
    }
    return *this;
  }
  void do_something()
#ifdef CHECK_VIRTUAL_STRUCTURE
    override
#endif
  {
  }
};

int main() {
  process_variadic_arg1<int>(1, 2, 3, 4, 5);
  process_variadic_arg2(1, 2, 3, 4, 5);

  std::cout << "is_trivial: " << std::is_trivial<std::string>::value << " is_standard_layout: " << std::is_standard_layout<std::string>::value << std::endl;
  std::cout << "is_trivial: " << std::is_trivial<A>::value << " is_standard_layout: " << std::is_standard_layout<A>::value << std::endl;
  return 0;
}
