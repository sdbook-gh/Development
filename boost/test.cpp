#include <iostream>
#include <boost/asio.hpp>
#include <functional>

class printer {
public:
  // 构造时引用 io_context 对象，使用它初始化 timer
  printer(boost::asio::io_context &ioc) : timer_(ioc, boost::asio::chrono::seconds(1)), count_(0) {
    timer_.async_wait(std::bind(&printer::print, this));
  }
  // 在析构中打印结果
  ~printer() { std::cout << "Final count is " << count_ << std::endl; }

  // 作为类的成员函数，无需再传入参数，直接使用当前对象的成员变量
  void print() {
    if (count_ < 5) {
      std::cout << count_ << std::endl;
      ++count_;
      timer_.expires_at(timer_.expiry() + boost::asio::chrono::seconds(1));
      timer_.async_wait(std::bind(&printer::print, this));
    }
  }
private:
  boost::asio::steady_timer timer_;
  int count_;
};
int main() {
  // main 里的调用简单了很多
  boost::asio::io_context io;
  printer p(io);
  io.run();
  return 0;
}
