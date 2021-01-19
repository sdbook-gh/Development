#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include "simple/NewProgressAction.h"
#include <iostream>

using ActionServer = actionlib::SimpleActionServer<simple::NewProgressAction>;
ActionServer *pServer = nullptr;

void maxStepSet()
{
    ROS_INFO("max and step set");
    auto pGoal = pServer->acceptNewGoal();
    int max = pGoal->max;
    int step = pGoal->step;
}

void startProgress()
{
    ROS_INFO("start progress");
    pServer->setPreempted();
    simple::NewProgressFeedback feedback;
    feedback.progress = 50;
    pServer->publishFeedback(feedback);
    simple::NewProgressResult result;
    result.result = 0;
    pServer->setSucceeded(result);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "NewProgress_Server");
  ros::NodeHandle nh;
  pServer = new ActionServer(nh, "NewProgress", false);
  pServer->registerGoalCallback(&maxStepSet);
  pServer->registerPreemptCallback(&startProgress);
  pServer->start();
  ROS_INFO("Ready to provide progress service");
  ros::spin();
  delete pServer;
  return 0;
}
