import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import socket
import cv2
import numpy as np
from cv_bridge import CvBridge

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

def main(args=None):
    rclpy.init(args=args)
    node = SocketCom("socket_com")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()