#include "ros/ros.h"
#include "simple/NewFunction.h"
#include <cstdlib>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "NewFunction_Client");
  ros::NodeHandle nh;
  ros::ServiceClient client = nh.serviceClient<simple::NewFunction>("NewFuncyion");
  simple::NewFunction srv;
  srv.request.request.content = "hello world";
  if (client.call(srv))
  {
    ROS_INFO("result: %d", srv.response.result);
  }
  else
  {
    ROS_ERROR("Failed to call service NewFuncyion");
    return -1;
  }
  return 0;
}
