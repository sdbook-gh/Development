#include "ros/ros.h"
#include "simple/NewFunction.h"

bool NewFunction(simple::NewFunction::Request &req, simple::NewFunction::Response &res)
{
  ROS_INFO("request: %s", req.request.content.c_str());
  res.result = 0;
  ROS_INFO("response: %d", res.result);
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "NewFunction_Server");
  ros::NodeHandle nh;
  ros::ServiceServer service = nh.advertiseService("NewFuncyion", NewFunction);
  ROS_INFO("Ready to provide service");
  ros::spin();
  return 0;
}
