#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "msgsrv_def/srv/custom.hpp"

void custom(const std::shared_ptr<msgsrv_def::srv::Custom::Request> request, std::shared_ptr<msgsrv_def::srv::Custom::Response> response)
{
  response->res = 0;
  RCLCPP_INFO(rclcpp::get_logger("service"), "service invoked Custom[%s] width[%d] height[%d]", request->req.header.frame_id.c_str(), request->req.width, request->req.height);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("custom_server");
  rclcpp::Service<msgsrv_def::srv::Custom>::SharedPtr service = node->create_service<msgsrv_def::srv::Custom>("Custom", &custom);
  RCLCPP_INFO(rclcpp::get_logger("service"), "Custom service ready");
  rclcpp::spin(node);
  rclcpp::shutdown();
}

