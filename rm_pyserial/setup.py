from setuptools import find_packages, setup

package_name = 'rm_pyserial'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ev3rm0re',
    maintainer_email='ev3rm0re@163.com',
    description='calculate the yaw and pitch of the target and send it to the serial port',
    license='GPL-3.0-only',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pyserial_node = rm_pyserial.pyserial_node:main'
        ],
    },
)
