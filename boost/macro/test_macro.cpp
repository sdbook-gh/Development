#include <boost/preprocessor.hpp>
#include <iostream>
#include <vector>

#define PRINT_MEMBER(_, ClassName, member)                                     \
  std::cout << #ClassName << "::"                                              \
            << BOOST_PP_STRINGIZE(member)                                      \
                                  << " offset: "                               \
                                  << offsetof(ClassName, member) << "\n";
#define PRINT_CLASS_MEMBERS(ClassName, ...)                                    \
  BOOST_PP_SEQ_FOR_EACH(PRINT_MEMBER, ClassName,                               \
                        BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

#define MACRO(_, data, elem) BOOST_PP_CAT(elem, data)
#define PROCESS(...)                                                           \
  BOOST_PP_STRINGIZE(BOOST_PP_SEQ_FOR_EACH(MACRO,,BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)))

#define MAKE_ENUM_AND_STRINGS_CALLBACK(r, data, elem) BOOST_PP_STRINGIZE(elem),
#define MAKE_ENUM_AND_STRINGS(...)                                             \
  enum test_enum { __VA_ARGS__ };                                              \
  std::vector<std::string> test_vector {                                       \
    BOOST_PP_LIST_FOR_EACH(MAKE_ENUM_AND_STRINGS_CALLBACK, ,                   \
                           BOOST_PP_VARIADIC_TO_LIST(__VA_ARGS__))             \
  }

int main() {
  struct MyClass {
    int a;
    double b;
    char c;
  };
  std::cout << PROCESS(1, 2, 3, 4) << "\n";
  MAKE_ENUM_AND_STRINGS(a, b, c, d);
  for (auto &&i : test_vector) {
    std::cout << i << '\n';
  }
  PRINT_CLASS_MEMBERS(MyClass, a, b, c);
  return 0;
}
