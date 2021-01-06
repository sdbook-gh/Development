#include <iostream>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "msgsrv_def/msg/custom.hpp"

class Subscriber : public rclcpp::Node
{
public:
  Subscriber()
  : Node("subscriber")
  {
    subscription_ = create_subscription<msgsrv_def::msg::Custom>(
      "/simple_topic",
      10,
      [&](msgsrv_def::msg::Custom::UniquePtr message) {
        RCLCPP_INFO(get_logger(), "Receive: Custom[%s] width[%d] height[%d]", message->header.frame_id.c_str(), message->width, message->height);
      });
  }

private:
  rclcpp::Subscription<msgsrv_def::msg::Custom>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  RCLCPP_INFO(rclcpp::get_logger("subscriber"), "Subscriber ready");
  rclcpp::spin(std::make_shared<Subscriber>());
  rclcpp::shutdown();
  return 0;
}

