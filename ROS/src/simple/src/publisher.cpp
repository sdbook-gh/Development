#include "ros/ros.h"
#include "simple/NewType.h"
#include <sstream>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "publisher");
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<simple::NewType>("topic", 100);
    ros::Rate loop_rate(10);

    int count = 0;
    while (ros::ok())
    {
        simple::NewType msg;
        std::stringstream ss;
        ss << "hello world " << count;
        msg.content = ss.str();
        ROS_INFO("%s", msg.content.c_str());
        pub.publish(msg);
        loop_rate.sleep();
        ++count;
    }
    return 0;
}
