#!/usr/bin/env python3
import serial
import struct
import time
import datetime
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def vrpnCallback(data):
    x = data.pose.position.x
    y = data.pose.position.y
    z = data.pose.position.z
    (roll, pitch, yaw) = euler_from_quaternion ([data.pose.orientation.x,data.pose.orientation.y,data.pose.orientation.z,data.pose.orientation.w])
    writeToXbee([x,y,z,yaw])
    rospy.loginfo(yaw)

def writeToXbee(data):
    # print(writeToXbee([1.1, 1.1, 1.1]))
    # data = [1.0, 1.0, 1.0, 1.0]
    f = len(data)
    # print(len(data))

    length = '22' # update this to work with any size message
    header = ['7E', '00', length, '10', '01', '00', '13', 'A2', '00', '41', 'B1', '91', '99', 'FF', 'FE', '00', '00','31','32']
    message = []
    for i in range(0, len(header)):
        message.append(bytearray.fromhex(header[i]))

    for i in range(len(data)):
        message.append(bytearray(struct.pack("f", data[i])))

    end = ['3C','3B']
    for i in range(0, len(end)):
        message.append(bytearray.fromhex(end[i]))

    output = bytearray()
    output = message[0]
    for i in range(1,len(message)):
        output = output + message[i]

    csum = sum(output[3:])
    low = 255 - (csum & 0xff)

    output = output + bytearray(low.to_bytes(1, 'big'))
    # print(low)
    # print(output.hex())

    # print('test writing')


    num = ser.write(output)

    # print(f'number of bytes written: {num}')


global ser
ser = serial.Serial("/dev/ttyUSB0",115200)
rospy.init_node('interface', anonymous=True)
rospy.Subscriber("/vrpn_client_node/chimera/pose", PoseStamped, vrpnCallback)

while not rospy.is_shutdown():
    rospy.spin()

ser.close()

