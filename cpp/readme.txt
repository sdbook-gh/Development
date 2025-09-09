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

# 变长类模板实例化过程
  tuple<int, double, char> three = tuple<int, double, char>(42, 42.0, 'a');
  get<2>(three);
  # tuple实例化过程
    template<typename T, typename ... Ts>
    struct tuple
    {
      inline tuple(const T & t, const Ts &... ts)
      : value(t)
      , rest(ts... )
      {
      }
      
      inline constexpr int size() const
      {
        return 1 + this->rest.size();
      }
      
      T value;
      tuple<Ts...> rest;
    };
    template<typename T>
    struct tuple<T>
    {
      inline tuple(const T & t)
      : value(t)
      {
      }
      
      inline constexpr int size() const
      {
        return 1;
      }
      
      T value;
    };
    /* First instantiated from: insights.cpp:43 */
    #ifdef INSIGHTS_USE_TEMPLATE
    template<>
    struct tuple<int, double, char>
    {
      inline tuple(const int & t, const double & __ts1, const char & __ts2)
      : value{t}
      , rest{tuple<double, char>(__ts1, __ts2)}
      {
      }
      
      inline constexpr int size() const;
      
      int value;
      tuple<double, char> rest;
    };

    #endif
    /* First instantiated from: insights.cpp:7 */
    #ifdef INSIGHTS_USE_TEMPLATE
    template<>
    struct tuple<double, char>
    {
      inline tuple(const double & t, const char & __ts1)
      : value{t}
      , rest{tuple<char>(__ts1)}
      {
      }
      
      inline constexpr int size() const;
      
      double value;
      tuple<char> rest;
    };

    #endif
    /* First instantiated from: insights.cpp:7 */
    #ifdef INSIGHTS_USE_TEMPLATE
    template<>
    struct tuple<char>
    {
      inline tuple(const char & t)
      : value{t}
      {
      }
      
      inline constexpr int size() const;
      
      char value;
    };

    #endif
  # nth_type实例化过程
    template<size_t N, typename T, typename ... Ts>
    struct nth_type : public nth_type<N - 1, Ts...>
    {
      
      /* PASSED: static_assert(N < (sizeof...(Ts) + 1), "index out of bounds"); */
    };
    template<typename T, typename ... Ts>
    struct nth_type<0, T, Ts...>
    {
      using value_type = T;
    };
    /* First instantiated from: insights.cpp:38 */
    #ifdef INSIGHTS_USE_TEMPLATE
    template<>
    struct nth_type<2, int, double, char> : public nth_type<1, double, char>
    {
      
      /* PASSED: static_assert(2UL < (2 + 1), "index out of bounds"); */
    };
    #endif
    /* First instantiated from: insights.cpp:16 */
    #ifdef INSIGHTS_USE_TEMPLATE
    template<>
    struct nth_type<1, double, char> : public nth_type<0, char>
    {
      
      /* PASSED: static_assert(1UL < (1 + 1), "index out of bounds"); */
    };
    #endif
    /* First instantiated from: insights.cpp:16 */
    #ifdef INSIGHTS_USE_TEMPLATE
    template<>
    struct nth_type<0, char>
    {
      using value_type = char;
    };
    #endif
  # getter实例化过程
    template<size_t N>
    struct getter
    {
      template<typename ... Ts>
      static inline typename nth_type<N, Ts...>::value_type & get(tuple<Ts...> & t)
      {
        return getter<N - 1>::get(t.rest);
      }
    };
    template<>
    struct getter<0>
    {
      template<typename T, typename ... Ts>
      static inline T & get(tuple<T, Ts...> & t)
      {
        return t.value;
      }
      
      /* First instantiated from: insights.cpp:27 */
      #ifdef INSIGHTS_USE_TEMPLATE
      template<>
      static inline char & get<char>(tuple<char> & t)
      {
        return t.value;
      }
      #endif
      
    };
    /* First instantiated from: insights.cpp:39 */
    #ifdef INSIGHTS_USE_TEMPLATE
    template<>
    struct getter<2>
    {
      template<typename ... Ts>
      static inline typename nth_type<2UL, Ts...>::value_type & get(tuple<Ts...> & t);
      
      /* First instantiated from: insights.cpp:39 */
      #ifdef INSIGHTS_USE_TEMPLATE
      template<>
      static inline typename nth_type<2UL, int, double, char>::value_type & get<int, double, char>(tuple<int, double, char> & t)
      {
        return getter<1>::get(t.rest);
      }
      #endif
      
    };
    #endif
    /* First instantiated from: insights.cpp:27 */
    #ifdef INSIGHTS_USE_TEMPLATE
    template<>
    struct getter<1>
    {
      template<typename ... Ts>
      static inline typename nth_type<1UL, Ts...>::value_type & get(tuple<Ts...> & t);
      
      /* First instantiated from: insights.cpp:27 */
      #ifdef INSIGHTS_USE_TEMPLATE
      template<>
      static inline typename nth_type<1UL, double, char>::value_type & get<double, char>(tuple<double, char> & t)
      {
        return getter<0>::get(t.rest);
      }
      #endif
      
    };
    #endif
  # get实例化过程
    template<size_t N, typename ... Ts>
    typename nth_type<N, Ts...>::value_type & get(tuple<Ts...> & t)
    {
      return getter<N>::get(t);
    }
    /* First instantiated from: insights.cpp:44 */
    #ifdef INSIGHTS_USE_TEMPLATE
    template<>
    typename nth_type<2UL, int, double, char>::value_type & get<2, int, double, char>(tuple<int, double, char> & t)
    {
      return getter<2>::get(t);
    }
    #endif
