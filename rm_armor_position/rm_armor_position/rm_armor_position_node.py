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
        self.device_name = self.declare_parameter('device_name', '/dev/ttyUSB0').get_parameter_value().string_value
        self.baud_rate = self.declare_parameter('baud_rate', 115200).get_parameter_value().integer_value
        self.parity = self.declare_parameter('parity', 'N').get_parameter_value().string_value
        self.stop_bits = float(self.declare_parameter('stop_bits', 1).get_parameter_value().integer_value)
        self.serialport = serialPort(self.device_name, self.baud_rate, self.parity, self.stop_bits)

    def tracker_info_callback(self, msg):
        x = msg.position.x
        y = msg.position.y
        z = msg.position.z
        yaw = math.atan2(y, x)
        pitch = math.atan2(z, x)
        self.get_logger().info('%f\n%f\r\n' % (yaw, pitch))
        self.serialport.send(b'%f\n%f\r\n' % (yaw, pitch))

    def __del__(self):
        self.serialport.close()
        self.get_logger().info('Serial port closed')

class serialPort:
    def __init__(self, port, baudrate, parity, stopbits):
        try:
            self.ser = serial.Serial(port=port, baudrate=baudrate,parity=parity, stopbits=stopbits)
        except serial.SerialException:
            self.ser = None
            print('Serial port open failed')

    def send(self, data):
        if self.ser is not None:
            self.ser.write(data)

    def read(self):
        if self.ser is not None:
            return self.ser.read()

    def close(self):
        if self.ser is not None:
            self.ser.close()

def main():
    rclpy.init()
    node = RmArmorPositionNode()
    try:
        rclpy.spin(node)
    except:
        node.destroy_node()