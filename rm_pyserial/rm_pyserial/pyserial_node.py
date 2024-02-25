import math
import serial
import struct
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from auto_aim_interfaces.msg import TrackerInfo

class PyserialNode(Node):
    def __init__(self):
        super().__init__('pyserial_node')
        self.count = 0
        self.yaws = []
        self.pitchs = []
        self.subscription = self.create_subscription(TrackerInfo, '/tracker/info', self.tracker_info_callback, 10)
        # 声明串口参数
        self.declare_parameter('device_name', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('parity', 'N')
        self.declare_parameter('stop_bits', 1)
        # 获取串口参数
        device_name = self.get_parameter('device_name').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        parity = self.get_parameter('parity').get_parameter_value().string_value
        stop_bits = self.get_parameter('stop_bits').get_parameter_value().integer_value
        # 串口初始化
        try:
            self.serialport = serialPort(device_name, baud_rate, parity, stop_bits)
        except serial.SerialException as e:
            self.get_logger().error('Serial port open failed: %s' % e)
            self.serialport = None
            sys.exit(1)

    def tracker_info_callback(self, msg):
        self.count += 1
        # 获取x,y,z
        x = msg.position.x
        y = msg.position.y
        z = msg.position.z
        # 计算yaw和pitch
        yaw = math.atan2(y, x)
        self.yaws.append(yaw)
        pitch = math.atan2(z, x)
        self.pitchs.append(pitch)
        if self.count % 6 == 0:
            # struct 16进制发送，大端模式，高位在前，填充53个字节，构成64字节的数据帧，后面还可以加其他数据
            data = b'\xA5' + struct.pack('>ff53x', np.mean(self.yaws), np.mean(self.pitchs)) + b'\r\n'
            self.get_logger().info('yaw: {}, pitch: {}.'.format(np.mean(self.yaws), np.mean(self.pitchs)))
            self.serialport.send(data)
            self.yaws.clear()
            self.pitchs.clear()
            self.count = 0

    def __del__(self):
        if self.serialport is not None:
            self.serialport.close()
            self.get_logger().info('Serial port closed')

class serialPort:
    def __init__(self, port, baudrate, parity, stopbits):
        # 打开串口
        self.ser = serial.Serial(port=port, baudrate=baudrate,parity=parity, stopbits=stopbits)

    def send(self, data):
        # 发送数据
        if self.ser is not None:
            self.ser.write(data)

    def read(self):
        # 读取数据
        if self.ser is not None:
            return self.ser.read()

    def close(self):
        # 关闭串口
        if self.ser is not None:
            self.ser.close()

def main():
    rclpy.init()
    node = PyserialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
            pass
    finally:
        node.destroy_node()