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

#define CHECK_VIRTUAL_STRUCTURE
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

template <typename T, size_t size>
inline size_t get_array_size(T (&array)[size]) noexcept {
  return size;
}

#include <vector>
template <typename T1, typename T2, template <typename> typename container = std::vector>
class TestTemplateTemplateParameter {
private:
  container<T1> c1;
  container<T2> c2;
};

template <size_t N>
struct string_literal {
  // constexpr string_literal(const char (&str)[N]) { std::copy_n(str, N, value); }
  constexpr string_literal(const char (&str)[N]) { std::copy_n(str, N, value); }
  char value[N];
};
template <string_literal str>
class TestTemplateStr {};

template <typename T>
struct foo {
  using value_type = T;
};
template <typename T, typename U = typename T::value_type>
struct bar {
  using value_type = U;
  int do_something();
};
template<typename T, typename U>
int bar<T, U>::do_something() {return 0;}

int main() {
  process_variadic_arg1<int>(1, 2, 3, 4, 5);
  process_variadic_arg2(1, 2, 3, 4, 5);
  std::cout << "is_trivial: " << std::is_trivial<std::string>::value << " is_standard_layout: " << std::is_standard_layout<std::string>::value << std::endl;
  std::cout << "is_trivial: " << std::is_trivial<A>::value << " is_standard_layout: " << std::is_standard_layout<A>::value << std::endl;

  int array[5] = {1, 2, 3, 4, 5};
  printf("array size: %ld\n", get_array_size(array));

  TestTemplateTemplateParameter<int, std::string> tttp;
  TestTemplateStr<"20"> tts;
  TestTemplateStr<string_literal<3>("20")> tts2;

  bar<foo<int>> x;
  return 0;
}
