#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <simple/NewProgressAction.h>

using ActionClient = actionlib::SimpleActionClient<simple::NewProgressAction>;
ActionClient *pClient = nullptr;

// Called once when the goal completes
void complete(const actionlib::SimpleClientGoalState& state,
            const simple::NewProgressResultConstPtr& result)
{
  ROS_INFO("complete");
  ros::shutdown();
}

// Called once when the goal becomes active
void active()
{
  ROS_INFO("active");
}

// Called every time feedback is received for the goal
void progress(const simple::NewProgressFeedbackConstPtr& feedback)
{
  ROS_INFO("progress");
}

int main (int argc, char **argv)
{
  ros::init(argc, argv, "NewProgress_Client");
  pClient = new ActionClient("NewProgress", true);
  ROS_INFO("Waiting for action server to start.");
  pClient->waitForServer();
  ROS_INFO("Action server started, sending goal.");
  simple::NewProgressGoal goal;
  goal.max = 100;
  goal.step = 10;
  pClient->sendGoal(goal, &complete, &active, &progress);
  ros::spin();
  delete pClient;
  return 0;
}
