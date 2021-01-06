import os
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class Sender(Node):

    def __init__(self):
        super().__init__('sender')
        self.publisher_ = self.create_publisher(Image, '/sy/cam/center/image', 10)
        timer_period = 0.5 # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        cvImage = cv2.imread('01.png', cv2.IMREAD_COLOR)
        if cvImage is None :
            print('cannot read image file')
            return
        cvBridge = CvBridge()
        imgMsg = cvBridge.cv2_to_imgmsg(cvImage)
        self.publisher_.publish(imgMsg)
        self.i += 1
        self.get_logger().info('sending image to /sy/cam/center/image')


def main(args=None):
    print('current working dir is ' + os.getcwd())
    rclpy.init(args=args)

    sender = Sender()

    rclpy.spin(sender)

    sender.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
