#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "msgsrv_def/msg/custom.hpp"

using namespace std::chrono_literals;

class Publisher : public rclcpp::Node
{
public:
  Publisher()
  : Node("publisher"), count_(0)
  {
    publisher_ = create_publisher<msgsrv_def::msg::Custom>("/simple_topic", 10);
    auto timer_callback =
      [&]() -> void {
        auto message = msgsrv_def::msg::Custom();
        message.header.frame_id = std::to_string(count_++);
        message.width = message.height = 1;
        message.data.push_back(0);
        RCLCPP_INFO(get_logger(), "Send: '%ld'", count_);
        publisher_->publish(message);
      };
    timer_ = create_wall_timer(500ms, timer_callback);
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<msgsrv_def::msg::Custom>::SharedPtr publisher_;
  size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  RCLCPP_INFO(rclcpp::get_logger("publisher"), "Publisher ready");
  rclcpp::spin(std::make_shared<Publisher>());
  rclcpp::shutdown();
  return 0;
}

