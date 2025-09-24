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

template <int N, int... M>
struct do_test : public do_test<N - 1, M...> {
  do_test() {
    std::cout << "do_test N: " << N << std::endl;
    //((std::cout << "do_test N: " << N << " M: " << M << " M size: " << sizeof...(M) << std::endl), ...);
  }
};
template <>
struct do_test<0> {
  do_test() { std::cout << "do_test_end" << std::endl; }
};

#include <tuple>
template <typename T, size_t... I>
void print_tuple(T const& tuple, std::index_sequence<I...>) {
  (..., (std::cout << std::get<I>(tuple) << " "));
  std::cout << std::endl;
  ((std::cout << std::get<I>(tuple) << " "), ...);
  std::cout << std::endl;
}
template <typename... T>
void print_tuple_new(const std::tuple<T...>& tuple) {
  [&]<typename TupType, size_t... I>(const TupType& tuple, std::index_sequence<I...>) { (..., (std::cout << std::get<I>(tuple) << " ")); }(tuple, std::make_index_sequence<sizeof...(T)>());
  std::cout << std::endl;
}

template <typename... Ts>
void print_rev(Ts&&... args) {
  (..., (std::cout << args << ' '));
  std::cout << std::endl;
  ((std::cout << args << ' '), ...);
  std::cout << std::endl;
}

template <typename T, size_t I>
T seq_to_val() {
  return (T)I;
}
template <typename T, size_t... I>
std::array<T, sizeof...(I)> make_array(std::index_sequence<I...>) {
  return std::array<T, sizeof...(I)>{seq_to_val<T, I>()...};
}

#include <array>
#include <utility>
#include <cstring>
struct Foo {
  int a;
  char b;
  float c;
};
// 把每个成员拷贝到缓冲区
template <std::size_t... I>
void serialize_impl(const Foo& f, unsigned char* buf, std::index_sequence<I...>) {
  // 用 lambda+下标 拿到第 I 个成员，再 memcpy
  auto copy = [&]<std::size_t J>(std::integral_constant<std::size_t, J>) {
    if constexpr (J == 0) std::memcpy(buf, &f.a, sizeof(f.a));
    if constexpr (J == 1) std::memcpy(buf, &f.b, sizeof(f.b));
    if constexpr (J == 2) std::memcpy(buf, &f.c, sizeof(f.c));
  };
  std::size_t offset = 0;
  ((copy(std::integral_constant<std::size_t, I>{}), offset += sizeof(std::tuple_element_t<I, std::tuple<int, char, float>>)), ...);
}
template <class... Members>
void serialize(const Foo& f, unsigned char* buf, std::index_sequence_for<Members...> seq) {
  serialize_impl(f, buf, seq);
}

template <typename T>
struct base_parser {
  void init() { std::cout << "init\n"; }
};
template <typename T>
struct parser : base_parser<T> {
  void parse() {
    this->init(); // OK
    base_parser<T>::init(); // OK
    std::cout << "parse\n";
  }
};

#include <map>
struct dictionary_traits {
  using key_type = int;
  using map_type = std::map<key_type, std::string>;
  static constexpr int identity = 1;
};
template <typename T>
struct dictionary : public T::map_type // [1]
{
  int start_key{T::identity}; // [2]
  typename T::key_type next_key; // [3]
};

template <typename T>
struct base_parser_N {
  template <typename U>
  struct token {};
  template <typename U>
  void init() {
    std::cout << "base_parser_N init\n";
  }
};
template <typename T>
struct parser_N : base_parser_N<T> {
  void parse() {
    using token_type = typename base_parser_N<T>::template token<int>; // [1]
    token_type t1{};
    typename base_parser_N<T>::template token<int> t2{}; // [2]
    std::cout << "parser_N parse\n";
    base_parser_N<T>::template init<T>();
  }
};

// CI Current Instantiation
template <typename T>
struct parser_CI {
  parser_CI* p1; // parser_CI is the CI
  parser_CI<T>* p2; // parser_CI<T> is the CI
  ::parser_CI<T>* p3; // ::parser_CI<T> is the CI
  parser_CI<T*> p4; // parser_CI<T*> is not the CI
  struct token {
    token* t1; // token is the CI
    parser_CI<T>::token* t2; // parser_CI<T>::token is the CI
    typename parser_CI<T*>::token* t3;
    // parser_CI<T*>::token is not the CI
  };
};
template <typename T>
struct parser_CI<T*> {
  parser_CI<T*>* p1; // parser_CI<T*> is the CI
  parser_CI<T>* p2; // parser_CI<T> is not the CI
};

template <unsigned int N>
constexpr unsigned int factorial = N * factorial<N - 1>;
template <>
constexpr unsigned int factorial<0> = 1;

#include <cxxabi.h>
template <typename T>
struct wrapper {};
template <int N>
struct manyfold_wrapper {
  using sub_type = manyfold_wrapper<N - 1>::value_type;
  using value_type = wrapper<sub_type>;
};
template <>
struct manyfold_wrapper<0> {
  using value_type = unsigned int;
};

void g(Foo& v) { std::cout << "g(foo&)\n"; }
void g(Foo&& v) { std::cout << "g(foo&&)\n"; }
void h(Foo& v) {
  // g(v);
  g(std::forward<Foo&>(v));
}
void h(Foo&& v) {
  //  g(v);
  g(std::forward<Foo&&>(v));
}
template <typename T>
void h_new(T&& v) {
  g(std::forward<T>(v));
}
#include <memory>
template <typename T, typename... Args>
std::unique_ptr<T> make_unique_new(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename Func, typename... Args>
auto syscall_with_check(Func func, Args&&... args) -> decltype(func(args...)) {
  auto result = func(std::forward<Args>(args)...);
  if (result == -1) {
    int saved_errno = errno;
    throw std::system_error(saved_errno, std::system_category(), "System call failed");
  }
  return result;
}

template <typename T>
struct wrapper_new;
template <typename T>
void print(wrapper_new<T> const& w);
template <typename T>
struct printer;
template <typename T>
struct wrapper_new {
  wrapper_new(T const v) : value(v) {}

private:
  T value;
  friend void print<T>(wrapper_new<T> const&);
  friend struct printer<void>;
};
template <typename T>
void print(wrapper_new<T> const& w) {
  std::cout << w.value << '\n'; /* error */
}
template <typename T>
struct printer {
  void operator()(wrapper_new<T> const& w) { std::cout << w.value << '\n'; /* error*/ }
};
template <>
struct printer<void> {
  template <typename T>
  void operator()(wrapper_new<T> const& w) {
    std::cout << w.value << '\n'; /* error*/
  }
};

template <typename T, size_t N>
void handle(T (&arr)[N], char (*)[N % 2 == 0 ? 1 : 0] = 0) {
  std::cout << "handle even array\n";
}
template <typename T, size_t N>
void handle(T (&arr)[N], char (*)[N % 2 == 1 ? 1 : 0] = 0) {
  std::cout << "handle odd array\n";
}

template <typename T>
struct foo_SFINAE {
  using foo_type = T;
};
template <typename T>
struct bar_SFINAE {
  using bar_type = T;
};
struct int_foo : foo_SFINAE<int> {};
struct int_bar : bar_SFINAE<int> {};
template <typename T>
decltype(typename T::foo_type(), void()) handle(T const& v) {
  std::cout << "handle a foo\n";
}
template <typename T>
decltype(typename T::bar_type(), void()) handle(T const& v) {
  std::cout << "handle a bar\n";
}
template <typename T>
decltype(static_cast<int>(T()), void()) handle(T const& v) {
  std::cout << "handle a number\n";
}

template <typename T, typename = typename std::enable_if_t<std::is_integral_v<T>>>
struct integral_wrapper {
  T value;
};
template <typename T, typename = typename std::enable_if_t<std::is_floating_point_v<T>>>
struct floating_wrapper {
  T value;
};

template <typename T>
bool are_equal(T const& a, T const& b) {
  if constexpr (std::is_floating_point_v<T>)
    return std::abs(a - b) < 0.001;
  else
    return a == b;
}

template <typename T>
std::string as_string(T value) {
  if constexpr (std::is_null_pointer_v<T>)
    return "null";
  else if constexpr (std::is_same_v<T, bool>)
    return value ? "true" : "false";
  else if constexpr (std::is_arithmetic_v<T>)
    return std::to_string(value);
}

namespace detail {
template <bool b>
struct copy_fn {
  template <typename InputIt, typename OutputIt>
  constexpr static OutputIt copy(InputIt first, InputIt last, OutputIt d_first) {
    while (first != last) { *d_first++ = *first++; }
    return d_first;
  }
};
template <>
struct copy_fn<true> {
  template <typename InputIt, typename OutputIt>
  constexpr static OutputIt* copy(InputIt* first, InputIt* last, OutputIt* d_first) {
    std::memmove(d_first, first, (last - first) * sizeof(InputIt));
    return d_first + (last - first);
  }
};
} // namespace detail
template <typename InputIt, typename OutputIt>
constexpr OutputIt my_copy(InputIt first, InputIt last, OutputIt d_first) {
  using input_type = std::remove_cv_t<typename std::iterator_traits<InputIt>::value_type>;
  using output_type = std::remove_cv_t<typename std::iterator_traits<OutputIt>::value_type>;
  constexpr bool opt = std::is_same_v<input_type, output_type> && std::is_pointer_v<InputIt> && std::is_pointer_v<OutputIt> && std::is_trivially_copy_assignable_v<input_type>;
  return detail::copy_fn<opt>::copy(first, last, d_first);
}

template <class T, T v>
struct integral_constant {
  static constexpr T value = v;
  using value_type = T;
};

template <typename... T>
void process_any(T&&... t) {
  ((std::cout << std::forward<T>(t) << "\n"), ...);
  (..., (std::cout << std::forward<T>(t) << "\n"));
}
// https://chat.deepseek.com/a/chat/s/49e0b9ad-61bf-41d7-8eb4-f49a51bd5f31
template <typename, typename... Ts>
struct has_common_type : std::false_type {};
template <typename... Ts>
struct has_common_type<std::void_t<std::common_type_t<Ts...>>, Ts...> : std::true_type {};
template <typename... Ts>
constexpr bool has_common_type_v = sizeof...(Ts) < 2 || has_common_type<void, Ts...>::value;
template <typename... Ts, typename = std::enable_if_t<has_common_type_v<Ts...>>>
void process(Ts&&... ts) {}

#include <concepts>
template <typename T>
  requires std::is_integral_v<T> && std::is_arithmetic_v<T>
T add(T const&& a, T const&& b) {
  return a + b;
}

template <typename T, typename U = void>
struct is_container : std::false_type {};
template <typename T>
struct is_container<T, std::void_t<typename T::value_type, typename T::size_type, typename T::allocator_type, typename T::iterator, typename T::const_iterator>> : std::true_type {};
template <typename T, typename U = void>
constexpr bool is_container_v = is_container<T, U>::value;
template <typename T>
concept container = requires(T t) {
  typename T::value_type;
  typename T::size_type;
  typename T::allocator_type;
  typename T::iterator;
  typename T::const_iterator;
  t.size();
  t.begin();
  t.end();
  t.cbegin();
  t.cend();
};
template <container T>
void process_container(T const& t) {
  std::cout << "container size: " << t.size() << std::endl;
}

template <typename T>
  requires std::is_arithmetic_v<T>
struct arithmetic_container {};
template <typename T>
concept arithmetic_container_concept = requires { typename arithmetic_container<T>; };

template <typename T, typename... Ts>
// inline constexpr bool are_same_v = std::conjunction_v<std::is_same<T, Ts>...>;
inline constexpr bool are_same_v = (std::is_same_v<T, Ts> && ...);
template <typename T, typename... Ts>
inline constexpr bool are_same_plus_v = (std::is_same_v<std::remove_cvref_t<T>, decltype(std::declval<T>() + std::declval<Ts>())> && ...);
// template <typename T, typename U>
// concept PlusResultIsSameAsT = requires(T t, U u) {
//   // 表达式 t + u 必须是合法的，该表达式的结果类型必须与 T 的纯净类型 (std::remove_cvref_t<T>) 相同。
//   { t + u } -> std::same_as<std::remove_cvref_t<T>>;
// };
// template <typename T, typename... Ts>
// inline constexpr bool are_same_plus_v = (PlusResultIsSameAsT<T, Ts> && ...);
template <typename... T>
concept HomogenousRange = requires(T... t) {
  (... + t);
  requires(are_same_plus_v<T...>);
  requires are_same_v<T...>;
  requires sizeof...(T) > 1;
};
template <typename... T>
  requires HomogenousRange<T...>
auto addmulti(T&&... t) {
  return (... + t);
}
template <typename... T>
  requires(std::is_integral_v<typename std::remove_reference_t<T>> && ...)
auto addmulti_atomic(T&&... args) {
  return (args++ + ...);
}

template <typename F, typename... T>
concept NonThrowing = requires(F&& func, T... t) {
  { func(t...) } -> std::convertible_to<int>;
  { func(t...) } noexcept;
  // requires std::is_convertible_v<decltype(func(t...)), int> && std::is_nothrow_invocable_v<F, T...>;
};
template <typename F, typename... T>
  requires NonThrowing<F, T...> && HomogenousRange<T...>
void invoke(F&& func, T... t) {
  func(t...);
}
template <typename T>
concept arithmatic = std::is_arithmetic_v<T>;
auto test_func(arithmatic auto a, arithmatic auto b) noexcept {
  printf("add: %lld, %lld\n", (long long int)a, (long long int)b);
  return a + b;
}
template <typename F, typename... T>
void invoke_new(F&& func, T&&... t) {
  std::forward<F>(func)(std::forward<T>(t)...);
}

#include <fcntl.h>
#include <chrono>
auto main() -> int {
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

  do_test<5>{};

  // 创建一个tuple并调用print_tuple打印
  // std::tuple<int, double, std::string> t{1, 2.5, "hello"};
  auto t{std::make_tuple(1, 2.5, "hello")};
  print_tuple<>(t, std::make_index_sequence<3>());
  print_tuple_new(t);

  print_rev(1, 2, 3);

  make_array<int>(std::make_index_sequence<5>());
  auto make_array_new = []<typename T, size_t... I>(std::index_sequence<I...>) { return std::array<T, sizeof...(I)>{I...}; };
  make_array_new.operator()<int>(std::make_index_sequence<5>());

  unsigned char buffer[sizeof(Foo)];
  Foo obj{123, 'X', 2.71f};
  serialize<int, char, float>(obj, buffer, {}); // 自动推导 3 个成员

  parser<int> pa;
  pa.parse();

  dictionary<dictionary_traits> d;

  parser_N<int> pa_N;
  pa_N.parse();

  std::cout << factorial<5> << '\n';

  std::cout << abi::__cxa_demangle(typeid(manyfold_wrapper<3>::value_type).name(), 0, 0, 0) << '\n';

  Foo f{1, '2', 3.0f};
  h(f);
  h(std::move(f));
  h_new(f);
  h_new(std::move(f));
  auto fup = make_unique_new<Foo>(f);
  fup = make_unique_new<Foo>(std::move(f));

  // syscall_with_check(open, "/tmp/test.txt", O_RDONLY);

  wrapper_new<int> w{43};
  print<int>(w);
  // print<char>(w);
  printer<void>()(w);
  // printer<int>()(w);

  int arr1[]{1, 2, 3, 4, 5};
  handle(arr1);
  int arr2[]{1, 2, 3, 4};
  handle(arr2);

  int_foo fi;
  int_bar bi;
  double ii = 0;
  handle(fi); // OK
  handle(bi); // OK
  handle(ii); // OK

  integral_wrapper<int>{.value = 1};
  floating_wrapper<double>{.value = 1.0};
  are_equal<double>(1, 1.0);

  std::vector<int> v1{1, 2, 3, 4, 5};
  std::vector<int> v2(5);
  // calls the generic implementation
  my_copy(std::begin(v1), std::end(v1), std::begin(v2));
  int a1[5] = {1, 2, 3, 4, 5};
  int a2[5];
  // calls the optimized implementation
  my_copy(a1, a1 + 5, a2);

  process(1, 2.0, 3.0);
  process_any(1, 2.0, "3");

  integral_constant<int, 1> vali;

  std::cout << "is_container_v<std::vector<int>>: " << is_container_v<std::vector<int>> << std::endl;
  process_container(v1);

  addmulti(1, 2, 3, 4, 5);
  addmulti_atomic(1, 2, 3, 4, 5);
  invoke(test_func<int, int>, 1, 2);
  int va = 1, vb = 2, vc = 3;
  // https://chat.deepseek.com/a/chat/s/6f38d309-9ef3-4227-9b4a-1cbf32e430ac
  invoke_new(addmulti<int&, int&, int&>, va, vb, vc); // addmulti类型为int& 参数类型折叠 int& && -> int&
  invoke_new(addmulti<int, int, int>, std::move(va), std::move(vb), std::move(vc)); // addmulti类型为int 参数类型为左值转换为 int&&
  invoke_new(addmulti<int, int, int>, 1, 2, 3); // addmulti类型为int 参数类型为右值 int&&
  invoke_new([](auto... args) { return addmulti(args...); }, va, vb, vc);
  addmulti_atomic(va, vb, vc);
  printf("%d %d %d\n", va, vb, vc);
  addmulti_atomic<int&, int&, int&>(va, vb, vc);
  printf("%d %d %d\n", va, vb, vc);
  invoke_new([](auto&&... args) { return addmulti_atomic(std::forward<decltype(args)>(args)...); }, va, vb, vc);
  printf("%d %d %d\n", va, vb, vc);

  auto lsum = [](std::integral auto a, std::integral auto b) { return a + b; };
  std::cout << lsum(1, 2) << std::endl;

  using std::string_literals::operator""s;
  return 0;
}
