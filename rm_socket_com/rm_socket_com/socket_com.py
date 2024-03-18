import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import socket
import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

class SocketCom(Node):
    def __init__(self, name):
        super().__init__(name)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('0.0.0.0', 1989))
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        print('Connected by', self.addr)
        self.sub = self.create_subscription(Image, '/detector/result_img', self.callback, 10)

    def callback(self, msg):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        _, img_encode = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_code = np.array(img_encode)
        img_data = img_code.tostring()
        self.conn.send(str(len(img_data)).ljust(16).encode())
        self.conn.send(img_data)

class VideoSave(Node):
    def __init__(self):
        super().__init__('video_save')
        self.sub = self.create_subscription(Image, '/image_raw', self.callback, qos_profile_sensor_data)
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('output.avi', self.fourcc, 30, (1280, 1024))

    def callback(self, msg):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self.out.write(img)

    def __del__(self):
        self.out.release()

def main(args=None):
    rclpy.init(args=args)
    # node = SocketCom("socket_com")
    # rclpy.spin(node)
    # node.destroy_node()
    # rclpy.shutdown()

    node = VideoSave()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
            pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()