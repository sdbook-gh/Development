#include <vector>
#include <cstdio>
#include <sstream>

template<typename T>
class MyQueue {
private:
    std::vector<T> _buffer;
    u_int32_t _head{0};
    u_int32_t _tail{0};
    u_int32_t _capacity;
public:
    explicit MyQueue(u_int32_t capacity): _buffer(capacity), _capacity(capacity) {}
    bool empty() {
        return _head == _tail;
    }
    bool full() {
        return (_tail + 1) % _capacity == _head;
    }
    bool push_back(const T& val) {
        if (full()) {
            printf("full\n");
            return false;
        }
        _buffer[_tail] = val;
        _tail = (_tail + 1) % _capacity;
        return true;
    }
    bool pop_front(T& val) {
        if (empty()) {
            printf("empty\n");
            return false;
        }
        val = _buffer[_head];
        _head = (_head + 1) % _capacity;
        return true;
    }
    std::string debug() {
        std::stringstream ss;
        ss << "_buffer [" << _buffer.size() << "] { ";
        for (auto &i : _buffer) {
            ss << i << " ";
        }
        ss << "} _head [" << _head << "] " << "_tail [" << _tail << "] ";
        return ss.str();
    }
};

int test_myqueue() {
    MyQueue<u_int32_t> myQueue(10);
    printf("1\n");
    printf("debug: %s\n", myQueue.debug().c_str());
    printf("empty: %d\n", myQueue.empty());
    printf("full: %d\n", myQueue.full());
    printf("2\n");
    myQueue.push_back(1);
    printf("debug: %s\n", myQueue.debug().c_str());
    u_int32_t val = 0;
    myQueue.pop_front(val);
    printf("debug: %s\n", myQueue.debug().c_str());
    printf("empty: %d\n", myQueue.empty());
    printf("full: %d\n", myQueue.full());
    myQueue.pop_front(val);
    printf("3\n");
    for (auto i = 0; i < 14; ++i) {
        myQueue.push_back(1);
        printf("debug: %s\n", myQueue.debug().c_str());
    }
    printf("4\n");
    for (auto i = 0; i < 14; ++i) {
        myQueue.pop_front(val);
        printf("debug: %s\n", myQueue.debug().c_str());
    }
    return 0;
}

#include <string>
auto & npos = std::string::npos;
bool reverse_string(std::string & in_str, size_t start_pos, size_t end_pos) {
    if (start_pos < 0) return false;
    if (start_pos == npos || end_pos == npos) return false;
    if (start_pos >= end_pos) return false;
    auto len = end_pos - start_pos;
    auto start_elem = &in_str[start_pos];
    auto end_elem = &in_str[end_pos];
    for (auto i = 0; i < len / 2; ++i) {
        auto val = start_elem[i];
        start_elem[i] = end_elem[- 1 - i];
        end_elem[- 1 - i] = val;
    }
    return true;
}
void reverse_full_string(std::string & in_str) {
    reverse_string(in_str, 0, in_str.length());
    size_t start_pos = 0;
    size_t end_pos = in_str.find(' ', start_pos);
    auto res = end_pos != npos;
    while(res) {
        res = reverse_string(in_str, start_pos, end_pos);
        if (res) {
            start_pos = end_pos + 1;
            end_pos = in_str.find(' ', start_pos);
            res = end_pos != npos;
        }
    }
    if (start_pos < in_str.length()) {
        reverse_string(in_str, start_pos, in_str.length());
    }
}
int test_reverse_string() {
    std::string str = "123";
    reverse_string(str, 0, str.length());
    printf("%s\n", str.c_str());
    str = "123 456";
    reverse_full_string(str);
    printf("%s\n", str.c_str());
    str = "123 456 7 890";
    reverse_full_string(str);
    printf("%s\n", str.c_str());
    return 0;
}

#include <algorithm>
#include <iostream>
template<typename T>
void print_vector(const std::vector<T>& v) {
    std::for_each(v.begin(), v.end(), [](const T& val) {
        std::cout << val << " ";
    });
    std::cout << std::endl;
}
#include <vector>
template<typename T>
std::vector<T> merge(const std::vector<T>& v1, const std::vector<T>& v2) {
    std::vector<T> v;
    auto pos1 = 0;
    auto pos2 = 0;
    for (; pos1 < v1.size() && pos2 < v2.size();) {
        if (v1[pos1] <= v2[pos2]) {
            v.emplace_back(v1[pos1++]);
        } else {
            v.emplace_back(v2[pos2++]);
        }
    }
    if (pos1 < v1.size()) {
        v.insert(v.end(), v1.begin() + pos1, v1.end());
    }
    if (pos2 < v2.size()) {
        v.insert(v.end(), v2.begin() + pos2, v2.end());
    }
    return v;
}
template<typename T>
std::vector<T> merge_new(const std::vector<T>& in_v, T in_val) {
    std::vector<T> v;
    std::vector<size_t> pos_v;
    for (auto i = 0; i < in_v.size(); ++i) {
        if (in_v[i] < in_val) {
            v.emplace_back(in_v[i]);
        } else {
            pos_v.emplace_back(i);
        }
    }
    for (auto i = 0; i < pos_v.size(); ++i) {
        v.emplace_back(in_v[pos_v[i]]);
    }
    return v;
}
int test_merge() {
    {
        std::vector<uint32_t> v1{1,2,3};
        std::vector<uint32_t> v2{4,5,6};
        auto v = merge<uint32_t>(v1, v2);
        print_vector<uint32_t>(v);
    }
    {
        std::vector<uint32_t> v1{1,2,3};
        std::vector<uint32_t> v2{1,2,3};
        auto v = merge<uint32_t>(v1, v2);
        print_vector<uint32_t>(v);
    }
    {
        std::vector<uint32_t> v{1,4,3,2,5,2};
        auto v_o = merge_new<uint32_t>(v, 3);
        print_vector<uint32_t>(v_o);
    }
    return 0;
}

#include <memory>
template<typename T>
class List;
template<typename T>
class Node {
private:
    std::shared_ptr<Node> _prev{nullptr};
    std::shared_ptr<Node> _next{nullptr};
    T _value;
    friend class List<T>;
};
template<typename T>
class List {
private:
    std::shared_ptr<Node<T>> _head{nullptr};
    std::shared_ptr<Node<T>> _tail{nullptr};
    size_t _size{0};
public:
    size_t size() {
        return _size;
    }
    void push_back(const T& val) {
        auto new_node = std::make_shared<Node<T>>();
        new_node->_value = val;
        if (_size == 0) {
            _head = _tail = new_node;
        } else {
            _tail->_next = new_node;
            new_node->_prev = _tail;
            _tail = new_node;
        }
        _size++;
    }
    bool pop_front(T& val) {
        if (_size == 0) return false;
        val = _head->_value;
        _head = _head->_next;
        if (_head == nullptr) {
            _tail = _head;
        } else {
            _head->_prev = nullptr;
        }
        _size--;
        return true;
    }
    void reverse() {
        auto node = std::make_shared<Node<T>>();
        auto e = _head;
        for (; e != nullptr;) {
            node = e->_prev;
            e->_prev = e->_next;
            e->_next = node;
            e = e->_prev;
        }
        node = _head;
        _head = _tail;
        _tail = node;
    }
    bool reverse_range(size_t begin, size_t end) {
        if (begin < 0) {
            printf("bad begin < 0\n");
        }
        if (begin > _size - 1) {
            printf("bad begin > _size - 1\n");
        }
        if (end < 0) {
            printf("bad end < 0\n");
        }
        if (end > _size - 1) {
            printf("bad end > _size - 1\n");
        }
        if (end < begin) {
            printf("bad end < begin\n");
        }
        auto e_b = _head;
        for (auto i = 0; i < begin; ++i) {
            e_b = e_b->_next;
        }
        auto e_e = e_b;
        for (auto i = begin; i < end; ++i) {
            e_e = e_e->_next;
        }
        auto head = e_b->_prev;
        auto tail = e_e->_next;
        auto node = std::make_shared<Node<T>>();
        for (auto e = e_b; e != tail;) {
            node = e->_next;
            e->_next = e->_prev;
            e->_prev = node;
            e = node;
        }
        if (head != nullptr) {
            head->_next = e_e;
        }
        e_e->_prev = head;
        if (tail != nullptr) {
            tail->_prev = e_b;
        }
        e_b->_next = tail;
        if (head == nullptr) {
            _head = e_e;
        }
        if (tail == nullptr) {
            _tail = e_b;
        }
        return true;
    }
    std::string debug() {
        std::stringstream ss;
        ss << "[" << _size << "] ";
        for (auto e = _head; e != nullptr; e = e->_next) {
            ss << e->_value << " ";
        }
        return ss.str();
    }
};
int test_list() {
    List<uint32_t> list;

    printf("%s\n", list.debug().c_str());
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    printf("%s\n", list.debug().c_str());
    uint32_t val = 0;
    list.pop_front(val);
    printf("%s\n", list.debug().c_str());
    list.pop_front(val);
    printf("%s\n", list.debug().c_str());
    list.pop_front(val);
    printf("%s\n", list.debug().c_str());
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    list.reverse();
    printf("%s\n", list.debug().c_str());
    list.push_back(4);
    list.push_back(5);
    list.push_back(6);
    printf("%s\n", list.debug().c_str());

    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    printf("%s\n", list.debug().c_str());
    list.reverse_range(0, 1);
    printf("%s\n", list.debug().c_str());
    list.reverse_range(0, list.size() - 1);
    printf("%s\n", list.debug().c_str());
    list.reverse_range(1, list.size() - 1);
    printf("%s\n", list.debug().c_str());
    return 0;
}

int main() {
    test_list();

    char key = 0;
    printf("press key to continue\n");
    std::cin >> key;

    std::vector<std::string> names = {"hi", "test", "foo"};
    std::vector<std::size_t> name_sizes;
    std::transform(names.begin(), names.end(), std::back_inserter(name_sizes), [](const std::string &name) {
        return name.size();
    });
    print_vector<std::size_t>(name_sizes);
}
