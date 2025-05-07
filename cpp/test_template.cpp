#include <initializer_list>
#include <iostream>

template <typename T, typename... Args>
void process_variadic_arg1(Args &&...args) {
  static_assert((std::is_convertible_v<Args, T> && ...),
                "All arguments must be convertible to T");
  std::initializer_list<T>{
      (std::cout << args << std::endl, T(std::forward<Args>(args)))...};
}

template <typename... Args> void process_variadic_arg2(Args &&...args) {
  std::initializer_list<int>{[&] {
    std::cout << args << std::endl;
    return 0;
  }()...};
}

int main() {
  process_variadic_arg1<int>(1, 2, 3, 4, 5);
  process_variadic_arg2(1, 2, 3, 4, 5);
  return 0;
}
