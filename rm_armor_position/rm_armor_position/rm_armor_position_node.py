import math
import serial
import rclpy
from rclpy.node import Node
from auto_aim_interfaces.msg import TrackerInfo

class RmArmorPositionNode(Node):
    def __init__(self):
        super().__init__('rm_armor_position_node')
        self.subscription = self.create_subscription(TrackerInfo, '/tracker/info', self.tracker_info_callback, 10)

        self.declare_parameter('device_name', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('parity', 'N')
        self.declare_parameter('stop_bits', 1)

        device_name = self.get_parameter('device_name').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        parity = self.get_parameter('parity').get_parameter_value().string_value
        stop_bits = self.get_parameter('stop_bits').get_parameter_value().integer_value
        
        self.serialport = serialPort(device_name, baud_rate, parity, stop_bits)

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
        except serial.SerialException as e:
            self.get_logger().error('Serial port open failed: %s' % e)
            self.ser = None

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