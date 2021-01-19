#include "ros/ros.h"
#include "simple/NewType.h"

void messageCallback(const simple::NewType::ConstPtr& msg)
{
    ROS_INFO("receive: [%s]", msg->content.c_str());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "subscriber");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("topic", 100, messageCallback);
    ros::spin();
    return 0;
}
