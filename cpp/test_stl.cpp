#include <vector>
#include <list>
#include <set>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <array>
#include <cassert>
#include <shared_mutex>
#include <mutex>
#include <map>
#include <ranges>
#include <format>

#if __cplusplus >= 202000L
template <typename T, std::size_t N>
  requires(N > 0)
class circular_buffer;
template <typename T, std::size_t N>
  requires(N > 0)
class circular_buffer_iterator {
public:
  using self_type = circular_buffer_iterator<T, N>;
  using value_type = T;
  using reference = value_type&;
  using const_reference = value_type const&;
  using pointer = value_type*;
  using const_pointer = value_type const*;
  using iterator_category = std::random_access_iterator_tag;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  explicit circular_buffer_iterator(circular_buffer<T, N>& buffer, size_type const index) : buffer_(buffer), index_(index) {}
  self_type& operator++() {
    if (index_ >= buffer_.get().size()) throw std::out_of_range("Iterator cannot be incremented past the end of the range");
    index_++;
    return *this;
  }
  self_type operator++(int) {
    self_type temp = *this;
    ++*this;
    return temp;
  }
  self_type& operator--() {
    if (index_ <= 0) throw std::out_of_range("Iterator cannot be decremented before the beginning of the range");
    index_--;
    return *this;
  }
  self_type operator--(int) {
    self_type temp = *this;
    --*this;
    return temp;
  }
  self_type operator+(difference_type offset) const {
    self_type temp = *this;
    return temp += offset;
  }
  self_type operator-(difference_type offset) const {
    self_type temp = *this;
    return temp -= offset;
  }
  self_type& operator+=(difference_type const offset) {
    difference_type next = (index_ + offset) % buffer_.get().capacity();
    if (next >= buffer_.get().size()) throw std::out_of_range("Iterator cannot be incremented past the bounds of the range");
    index_ = next;
    return *this;
  }
  self_type& operator-=(difference_type const offset) { return *this += -offset; }
  bool operator==(self_type const& other) const { return compatible(other) && index_ == other.index_; }
  bool operator!=(self_type const& other) const { return !(*this == other); }
  bool operator<(self_type const& other) const { return index_ < other.index_; }
  bool operator>(self_type const& other) const { return other < *this; }
  bool operator<=(self_type const& other) const { return !(other < *this); }
  bool operator>=(self_type const& other) const { return !(*this < other); }
  const_reference operator*() const {
    if (buffer_.get().empty() || !in_bounds()) throw std::logic_error("Cannot dereferentiate the iterator");
    return buffer_.get().data_[(buffer_.get().head_ + index_) % buffer_.get().capacity()];
  }
  const_reference operator->() const {
    if (buffer_.get().empty() || !in_bounds()) throw std::logic_error("Cannot dereferentiate the iterator");
    return buffer_.get().data_[(buffer_.get().head_ + index_) % buffer_.get().capacity()];
  }
  reference operator*() {
    if (buffer_.get().empty() || !in_bounds()) throw std::logic_error("Cannot dereferentiate the iterator");
    return buffer_.get().data_[(buffer_.get().head_ + index_) % buffer_.get().capacity()];
  }
  reference operator->() {
    if (buffer_.get().empty() || !in_bounds()) throw std::logic_error("Cannot dereferentiate the iterator");
    return buffer_.get().data_[(buffer_.get().head_ + index_) % buffer_.get().capacity()];
  }
  value_type& operator[](difference_type const offset) { return *((*this + offset)); }
  value_type const& operator[](difference_type const offset) const { return *((*this + offset)); }

private:
  std::reference_wrapper<circular_buffer<T, N>> buffer_;
  size_type index_ = 0;
  bool compatible(self_type const& other) const { return buffer_.get().data_.data() == other.buffer_.get().data_.data(); }
  bool in_bounds() const {
    if (index_ >= buffer_.get().size()) return false;
    return true;
  }
};
template <typename T, std::size_t N>
  requires(N > 0)
class circular_buffer {
public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = value_type const&;
  using pointer = value_type*;
  using const_pointer = value_type const*;
  using iterator = circular_buffer_iterator<T, N>;
  using const_iterator = circular_buffer_iterator<T const, N>;

  constexpr circular_buffer() = default;
  constexpr circular_buffer(value_type const (&values)[N]) : size_(N), tail_(N - 1) { std::copy(std::begin(values), std::end(values), data_.begin()); }
  constexpr circular_buffer(const_reference v) : size_(N), tail_(N - 1) { std::fill(data_.begin(), data_.end(), v); }

  constexpr size_type size() const noexcept { return size_; }
  constexpr size_type capacity() const noexcept { return N; }
  constexpr bool empty() const noexcept { return size_ == 0; }
  constexpr bool full() const noexcept { return size_ == N; }
  constexpr void clear() noexcept {
    size_ = 0;
    head_ = 0;
    tail_ = 0;
  };
  constexpr reference operator[](size_type const pos) { return data_[(head_ + pos) % N]; }
  constexpr const_reference operator[](size_type const pos) const { return data_[(head_ + pos) % N]; }
  constexpr reference at(size_type const pos) {
    if (pos < size_) return data_[(head_ + pos) % N];
    throw std::out_of_range("Index is out of range");
  }
  constexpr const_reference at(size_type const pos) const {
    if (pos < size_) return data_[(head_ + pos) % N];
    throw std::out_of_range("Index is out of range");
  }
  constexpr reference front() {
    if (size_ > 0) return data_[head_];
    throw std::logic_error("Buffer is empty");
  }
  constexpr const_reference front() const {
    if (size_ > 0) return data_[head_];
    throw std::logic_error("Buffer is empty");
  }
  constexpr reference back() {
    if (size_ > 0) return data_[tail_];
    throw std::logic_error("Buffer is empty");
  }
  constexpr const_reference back() const {
    if (size_ > 0) return data_[tail_];
    throw std::logic_error("Buffer is empty");
  }
  constexpr void push_back(T const& value) {
    if (empty()) {
      data_[tail_] = value;
      size_++;
    } else if (!full()) {
      data_[++tail_] = value;
      size_++;
    } else {
      head_ = (head_ + 1) % N;
      tail_ = (tail_ + 1) % N;
      data_[tail_] = value;
    }
  }
  constexpr void push_back(T&& value) {
    if (empty()) {
      data_[tail_] = std::move(value);
      size_++;
    } else if (!full()) {
      data_[++tail_] = std::move(value);
      size_++;
    } else {
      head_ = (head_ + 1) % N;
      tail_ = (tail_ + 1) % N;
      data_[tail_] = std::move(value);
    }
  }
  constexpr T pop_front() {
    if (empty()) throw std::logic_error("Buffer is empty");
    size_type index = head_;
    head_ = (head_ + 1) % N;
    size_--;
    return data_[index];
  }
  iterator begin() { return iterator(*this, 0); }
  iterator end() { return iterator(*this, size_); }
  const_iterator begin() const { return const_iterator(*this, 0); }
  const_iterator end() const { return const_iterator(*this, size_); }

private:
  std::array<value_type, N> data_;
  size_type head_ = 0;
  size_type tail_ = 0;
  size_type size_ = 0;
  friend circular_buffer_iterator<T, N>;
};

template <typename InputIt1, typename InputIt2, typename OutputIt>
  requires requires(InputIt1 it1, InputIt2 it2, OutputIt out) {
    *out++ = *it1++;
    *out++ = *it2++;
    // 条件1: InputIt1 必须满足 std::input_iterator 概念
    typename std::iterator_traits<InputIt1>::value_type;
    requires std::input_iterator<InputIt1>;
    // 条件2: InputIt1 和 InputIt2 必须是相同类型
    requires std::same_as<InputIt1, InputIt2>;
    // 条件3: OutputIt 必须是可接收 InputIt1 元素类型的 back_inserter 类型
    typename OutputIt::container_type;
    typename OutputIt::container_type::value_type;
    requires std::is_convertible_v<typename std::iterator_traits<InputIt1>::value_type, typename OutputIt::container_type::value_type>;
    requires sizeof(typename OutputIt::container_type::value_type) >= sizeof(typename std::iterator_traits<InputIt1>::value_type);
  }
OutputIt flatzip(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt dest) {
  auto it1 = first1;
  auto it2 = first2;
  while (it1 != last1 && it2 != last2) {
    *dest++ = *it1++;
    *dest++ = *it2++;
  }
  return dest;
}

template <typename K, typename V>
class ThreadSafeMap {
private:
  std::map<K, V> map;
  mutable std::shared_mutex mutex;

public:
  V get(const K& key) {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return map[key];
  }

  void set(const K& key, const V& value) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    map[key] = value;
  }
};

template <typename Func>
struct range_adapter {
  Func func;
  // 构造函数
  constexpr range_adapter(Func f) : func(std::move(f)) {}
  // 重载 operator| 使其可以用于管道语法
  template <std::ranges::viewable_range Range>
  friend constexpr auto operator|(Range&& range, const range_adapter& adapter) {
    return adapter.func(std::forward<Range>(range));
  }
};
// 辅助函数：将 lambda 包装成适配器
template <typename Func>
constexpr auto make_range_adapter(Func&& func) {
  return range_adapter<std::decay_t<Func>>{std::forward<Func>(func)};
}

int main() {
  std::vector<int> v{1, 2, 3};
  // copy vector to vector
  std::vector<int> vc(v.size());
  std::copy(v.begin(), v.end(), vc.begin()); // like memcpy
  std::cout << "vc:\n";
  std::copy(vc.begin(), vc.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\n";
  // copy vector to list
  std::list<int> l;
  std::copy(v.begin(), v.end(), std::back_inserter(l)); // like push_back
  std::cout << "l:\n";
  std::copy(l.begin(), l.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\n";
  // copy list to set
  std::set<int> s;
  std::copy(l.begin(), l.end(), std::inserter(s, s.begin())); // like insert
  std::cout << "s:\n";
  std::copy(s.begin(), s.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\n";

  circular_buffer b({1, 2, 3, 4});
  assert(b.size() == 4);
  b.push_back(5);
  b.push_back(6);
  b.pop_front();
  assert(b.size() == 3);
  assert(b[0] == 4);
  assert(b[1] == 5);
  assert(b[2] == 6);
  std::vector<decltype(b)::value_type> vv;
  for (auto it = b.begin(); it != b.end(); ++it) { vv.push_back(*it); }
  std::cout << "vv:\n";
  std::copy(vv.begin(), vv.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\n";
  *b.begin() = 0;
  assert(b[0] == 0);
  std::cout << "b:\n";
  for (auto& elem : b) { std::cout << elem << " "; }
  std::cout << "\n";

  {
    std::vector<int> v1{1, 2, 3};
    std::vector<int> v2;
    std::vector<int> v3;
    flatzip(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(v3));
    assert(v3.empty());
  }
  {
    std::vector<int> v1{100, 200, 300};
    std::vector<int> v2{400, 500};
    std::vector<long> v3;
    flatzip(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(v3));
    assert(v3 == decltype(v3)({100, 400, 200, 500}));
  }

  ThreadSafeMap<int, int> map;
  map.set(1, 100);
  assert(map.get(1) == 100);

  {
    std::vector<long> v{999, 888, 777};
    std::ranges::sort(v, std::less<>());
    auto r = v | std::ranges::views::transform([](auto x) { return x + 0; }) | std::ranges::views::drop(1) | std::ranges::views::take(1) | std::ranges::views::transform([](auto x) {
               std::cout << x << " ";
               return x;
             });
    std::ranges::for_each(r, [](auto x) {});
    std::cout << "\n";
  }
  {
    namespace rv = std::ranges::views;
    std::vector<std::tuple<int, double, std::string>> v = {{1, 1.1, "one"}, {2, 2.2, "two"}, {3, 3.3, "three"}};
    for (auto i : v | rv::keys) std::cout << i << '\n'; // prints 1 2 3
    for (auto i : v | rv::values) std::cout << i << '\n'; // prints 1.1 2.2 3.3
    for (auto i : v | rv::elements<2>) std::cout << i << '\n'; // prints one two three
    {
#if __cplusplus >= 202300L
      std::vector<std::vector<int>> v{{1, 2, 3}, {4}, {5, 6}};
      for (int const i : v | rv::join_with(0)) std::cout << i << ' '; // print 1 2 3 0 4 0 5 6
      std::cout << "\n";
#endif
    }
    std::string text{"this is a demo!"};
    std::string_view delim{" "};
    for (auto const word : text | rv::split(delim)) { std::cout << std::string_view(word.begin(), word.end()) << '\n'; }
    {
#if __cplusplus >= 202300L
      std::array<int, 4> a{1, 2, 3, 4};
      std::vector<double> v{10.1f, 20.2f, 30.3f};
      for (auto const [i, j] : rv::zip(a, v)) std::cout << std::format("({}, {})\n", i, j); // print 1 10.0 2 20.0 3 30.0
#endif
    }
    {
#if __cplusplus >= 202300L
      std::vector<int> v{1, 2, 3, 4};
      for (auto i : v | rv::adjacent<3>) { std::cout << std::format("({},{},{})\n", std::get<0>(i), std::get<1>(i), std::get<2>(i)); }
      for (auto i : v | rv::adjacent_transform<2>(std::multiplies())) { std::cout << i << ' '; } // prints: 3 24 60
      std::cout << "\n";
#endif
    }
    {
      struct Item {
        int id;
        std::string name;
        double price;
      };
      std::vector<Item> items{{1, "pen", 5.49}, {2, "ruler", 3.99}, {3, "pensil case", 12.50}};
      std::vector<Item> copies;
      std::ranges::copy_if(items, std::back_inserter(copies), [](auto const& name) { return name[0] == 'p'; }, &Item::name);
      std::vector<std::string> names;
      std::ranges::copy_if(items | rv::transform(&Item::name), std::back_inserter(names), [](std::string const& name) { return name[0] == 'p'; });
    }
  }

  // 自定义的range adapter：过滤偶数
  auto even_filter = make_range_adapter([](auto&& range) { return range | std::views::filter([](auto& x) { return x % 2 == 0; }); });
  // 自定义的range adapter：平方转换
  auto square_transform = make_range_adapter([](auto&& range) { return range | std::views::transform([](auto& x) { return x * x; }); });
  std::vector<int> numbers{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  // 使用自定义的range adapter
  auto result = numbers | even_filter | square_transform;
  std::ranges::for_each(result, [](auto x) { std::cout << x << " "; });
  std::cout << "\n";

  return 0;
}
#else
int main() { return 0; }
#endif
