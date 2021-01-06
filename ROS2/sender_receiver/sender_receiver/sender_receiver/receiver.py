import tensorflow as tf
import os
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import cv2
import numpy as np
import sender_receiver.lanenet_postprocess_new
from sensor_msgs.msg import Image
from ados_lane_msgs.msg import ImageRoiBitmap
from array import array
import datetime


class Receiver(Node):

    def __init__(self):
        super().__init__('Receiver')
        self.publisher = self.create_publisher(ImageRoiBitmap, '/cv/lane_detection/cnn_output', 10)
        self.subscription = self.create_subscription(Image, '/sy/cam/center/image', self.listener_callback, 10)
        self.subscription # prevent unused variable warning
        self.graph = tf.Graph()
        self.return_elements = ['input_tensor:0', 'final_binary_output:0', 'final_pixel_embedding_output:0']
        self.return_tensors = self.read_pb_return_tensors(self.graph, 'frozen_model.pb', self.return_elements)
        self.sess = tf.Session(graph = self.graph)

    def __del__(self):
        self.sess.close()

    def listener_callback(self, msg):
        self.get_logger().info('receive Image frame_id: %s encoding: %s' % (msg.header.frame_id, msg.encoding))
        cvBridge = CvBridge()
        cvImage = cvBridge.imgmsg_to_cv2(msg, msg.encoding)
        data = self.pb_predict(cvImage)
        bitMapMsg = ImageRoiBitmap()
        bitMapMsg.header.frame_id = msg.header.frame_id
        bitMapMsg.orig_height = msg.height
        bitMapMsg.orig_width = msg.width
        bitMapMsg.resized_height = 256
        bitMapMsg.resized_width = 512
        bitMapMsg.bitmap = data
        self.publisher.publish(bitMapMsg)
        self.get_logger().info('send image to /sy/cam/center/image_processed')

    def image_preporcess(self, image, target_size):
        imageResize = cv2.resize(image, target_size, interpolation = cv2.INTER_AREA)
        image = imageResize / 127.5 - 1.0
        return imageResize, image

    def read_pb_return_tensors(self, graph, pb_file, return_elements):
        with tf.gfile.FastGFile(pb_file, 'rb') as f:
            frozen_graph_def = tf.compat.v1.GraphDef()
            frozen_graph_def.ParseFromString(f.read())
        with graph.as_default():
            return_elements = tf.import_graph_def(frozen_graph_def, return_elements = return_elements)
        return return_elements

    def pb_predict(self, image):
        dt_ms = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
        #cv2.imwrite('debug/' + 'orig_' + dt_ms + '.png', image)
        resizedImage, processedImage = self.image_preporcess(image, (512, 256))
        #cv2.imwrite('debug/' + 'resized_' + dt_ms + '.png', resizedImage)
        #cv2.imwrite('debug/' + 'processed_' + dt_ms + '.png', processedImage)
        try:
            binary, embedding = self.sess.run([self.return_tensors[1], self.return_tensors[2]], feed_dict = {self.return_tensors[0] : [processedImage]})
            binary *= 67
            cv2.imwrite('debug/' + 'binary_' + dt_ms + '.png', binary)
            #[height, width] = binary.shape
            #print('height=' + str(height) + ' cols=' + str(width) + ' type=' + str(type(binary)) + '-' + str(binary.dtype))
            returnData = array('B', binary.astype('uint8').flatten().tolist())
        finally:
            print('pb_predict complete successfully')
        return returnData


def main(args=None):
    print('current working dir is ' + os.getcwd())
    rclpy.init(args=args)

    receiver = Receiver()

    rclpy.spin(receiver)

    receiver.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
