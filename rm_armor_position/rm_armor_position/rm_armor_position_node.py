import math
import serial
import rclpy
from rclpy.node import Node
from auto_aim_interfaces.msg import TrackerInfo

class RmArmorPositionNode(Node):
    def __init__(self):
        super().__init__('rm_armor_position_node')
        self.subscription = self.create_subscription(
            TrackerInfo,
            '/tracker/info',
            self.tracker_info_callback,
            10)
        self.serialport = serialPort('/dev/ttyUSB0', 115200)

    def tracker_info_callback(self, msg):
        x = msg.position.x
        y = msg.position.y
        z = msg.position.z
        yaw = math.atan2(y, x)
        pitch = math.atan2(z, x)
        self.get_logger().info('%f\n%f\r\n' % (yaw, pitch))
        self.serialport.send(b'%f\n%f\r\n' % (yaw, pitch))

    def __del__(self):
        self.get_logger().info('Closing serial port...')
        self.serialport.close()

class serialPort:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(port, baudrate)

    def send(self, data):
        self.ser.write(data)

    def read(self):
        return self.ser.read()

    def close(self):
        self.ser.close()

def main():
    rclpy.init()
    node = RmArmorPositionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        rclpy.shutdown()

if __name__ == '__main__':
    main()