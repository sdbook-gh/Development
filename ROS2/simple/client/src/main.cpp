#include "rclcpp/rclcpp.hpp"
#include "msgsrv_def/srv/custom.hpp"

#include <chrono>
#include <cstdlib>
#include <memory>

using namespace std::chrono_literals;

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("custom_client");
  rclcpp::Client<msgsrv_def::srv::Custom>::SharedPtr client = node->create_client<msgsrv_def::srv::Custom>("Custom");
  while (!client->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(rclcpp::get_logger("client"), "Custom client failed");
      return -1;
    }
    RCLCPP_INFO(rclcpp::get_logger("client"), "wait for Custom service ready");
  }
  RCLCPP_INFO(rclcpp::get_logger("client"), "Custom client ready");
  while(true)
  {
    auto request = std::make_shared<msgsrv_def::srv::Custom::Request>();
    request->req.header.frame_id = "1";
    request->req.width = request->req.height = 1;
    request->req.data.push_back(0);
    auto result = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(node, result) == rclcpp::executor::FutureReturnCode::SUCCESS)
    {
      RCLCPP_INFO(rclcpp::get_logger("client"), "invoke Custom service result: %d", result.get()->res);
    } else {
      RCLCPP_ERROR(rclcpp::get_logger("client"), "invoke Custom service failed");
      break;
    }
  }
  rclcpp::shutdown();
  return 0;
}

