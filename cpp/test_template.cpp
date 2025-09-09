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
template <typename T, typename U>
int bar<T, U>::do_something() {
  return 0;
}

#include <type_traits>

// 1. 主模板：对任意类型 T 都成立
template <typename T>
struct TypeDescr {
  static void print() { std::cout << "generic T\n"; }
};
// 2. 偏特化 1：当实参是指针类型时
template <typename T>
struct TypeDescr<T*> {
  static void print() { std::cout << "pointer to " << (std::is_const_v<T> ? "const " : "") << "T\n"; }
};
// 3. 偏特化 2：当实参是数组类型时（任意维）
template <typename T, std::size_t N>
struct TypeDescr<T[N]> {
  static void print() { std::cout << "array of " << N << " T's\n"; }
};

template <typename T>
constexpr T SEPARATOR = '\n';
template <>
constexpr wchar_t SEPARATOR<wchar_t> = L'\n';
template <typename T>
struct is_floating_point {
  constexpr static bool value = false;
};
template <>
struct is_floating_point<float> {
  constexpr static bool value = true;
};

#include <typeinfo>
template <typename T, size_t S>
struct list {
  using type = std::vector<T>;
};
template <typename T>
struct list<T, 1> {
  using type = T;
};
template <typename T, size_t S>
using list_t = typename list<T, S>::type;

#include <functional>
template <typename, typename>
struct func_pair;
template <typename R1, typename... A1, typename R2, typename... A2>
struct func_pair<R1(A1...), R2(A2...)> {
  std::function<R1(A1...)> f;
  std::function<R2(A2...)> g;
};

template <typename T, typename... Ts>
struct tuple {
  tuple(T const& t, Ts const&... ts) : value(t), rest(ts...) {}
  constexpr int size() const { return 1 + rest.size(); }
  T value;
  tuple<Ts...> rest;
};
template <typename T>
struct tuple<T> {
  tuple(const T& t) : value(t) {}
  constexpr int size() const { return 1; }
  T value;
};
template <size_t N, typename T, typename... Ts>
struct nth_type : nth_type<N - 1, Ts...> {
  static_assert(N < sizeof...(Ts) + 1, "index out of bounds");
};
template <typename T, typename... Ts>
struct nth_type<0, T, Ts...> {
  using value_type = T;
};
template <size_t N>
struct getter {
  template <typename... Ts>
  static typename nth_type<N, Ts...>::value_type& get(tuple<Ts...>& t) {
    return getter<N - 1>::get(t.rest);
  }
};
template <>
struct getter<0> {
  template <typename T, typename... Ts>
  static T& get(tuple<T, Ts...>& t) {
    return t.value;
  }
};
template <size_t N, typename... Ts>
typename nth_type<N, Ts...>::value_type& get(tuple<Ts...>& t) {
  return getter<N>::get(t);
}

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

  int a{};
  int* p{};
  int arr[5]{};
  const int* cp{};
  TypeDescr<decltype(a)>::print();
  TypeDescr<decltype(p)>::print();
  TypeDescr<decltype(arr)>::print();
  TypeDescr<decltype(cp)>::print();

  printf("%d\n", SEPARATOR<char>);
  printf("%d\n", SEPARATOR<wchar_t>);
  printf("%d\n", is_floating_point<float>::value);
  printf("%d\n", is_floating_point<int>::value);
  printf("%d\n", is_floating_point<float>::value);

  printf("%s\n", typeid(list_t<int, 1>).name());
  printf("%s\n", typeid(list_t<int, 2>).name());

  func_pair<void(), void()> fp;
  fp.f = []() { std::cout << "hello" << std::endl; };
  fp.g = []() { std::cout << "world" << std::endl; };
  fp.f();
  fp.g();

  tuple<int, double, char> three(42, 42.0, 'a');
  std::cout << get<0>(three) << std::endl;
  return 0;
}
