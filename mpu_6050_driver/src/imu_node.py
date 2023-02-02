#!/usr/bin/env python2.7

import time
import smbus
import struct
import rospy
import numpy as np
from sensor_msgs.msg import Temperature, Imu
from tf.transformations import quaternion_about_axis
from mpu_6050_driver.registers import PWR_MGMT_1, ACCEL_XOUT_H, ACCEL_YOUT_H, ACCEL_ZOUT_H, TEMP_H,\
    GYRO_XOUT_H, GYRO_YOUT_H, GYRO_ZOUT_H, ACCEL_ZOUT_L, GYRO_ZOUT_L
import sys
sys.path.append('../../')
from mini_ros.msg import IMUdata

ADDR = None
bus = None
IMU_FRAME = None

DISCON = False



import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)



# read_word and read_word_2c from http://blog.bitify.co.uk/2013/11/reading-data-from-mpu-6050-on-raspberry.html


def read_word_BLOCK(adr_start):
    """The original one that crashes"""
    while True:
        try:
            with time_limit(1):
                # print("trying registers")
                Register = bus.read_i2c_block_data(ADDR, adr_start,6)
                # print("finished reading registers")
                # print("number of registers:", len(Register))
                vallist = []
                for i in range(3):
                    j = i*2
                    high, low = Register[j:j+2]
                    val = (high << 8) + low
                    # return val
                    if (val >= 0x8000):
                        newval =  -((65535 - val) + 1)
                    else:
                        newval = val
                    vallist.append(newval)
                return np.array(vallist)
        except:
            try:
                # print("Discon")
                bus = smbus.SMBus(1)
                ADDR = rospy.get_param('~device_address', 0x68)
                if type(ADDR) == str:
                    ADDR = int(ADDR, 16)
                bus.write_byte_data(ADDR, PWR_MGMT_1,0 )
                # print("End Discon")
            except:
                print("couldn't even reconnect to bus")



def publish_imu_spot():
    starttime1 = time.time()

    # print("read ACC")
    accel_x, accel_y, accel_z = read_word_BLOCK(ACCEL_XOUT_H)/ 16384.0
    # print("finished ACC_z")

    # Calculate a quaternion representing the orientation
    accel = accel_x, accel_y, accel_z
    ref = np.array([0, 0, 1])
    acceln = accel / np.linalg.norm(accel)
    axis = np.cross(acceln, ref)
    angle = np.arccos(np.dot(acceln, ref))
    orientation = quaternion_about_axis(angle, axis)
    orx, ory, orz, orw = orientation

    # print("read Gyr")
    gyro_x, gyro_y, gyro_z = read_word_BLOCK(GYRO_XOUT_H) /  131.0
    # print("finished GYR_Z")
    
    imudata = IMUdata()
    imudata.roll = orx
    imudata.pitch = ory
    imudata.acc_x = accel_x
    imudata.acc_y = accel_y
    imudata.acc_z = accel_z
    imudata.gyro_x = gyro_x
    imudata.gyro_y = gyro_y
    imudata.gyro_z = gyro_z

    imu_pub_spot.publish(imudata)
    # print("imu took:", time.time() - starttime1)


temp_pub = None
imu_pub = None
imu_pub_spot = None

if __name__ == '__main__':
    rospy.init_node('imu_node')
    print(" params says its:", rospy.get_param('~bus', 1))
    bus = smbus.SMBus(1)
    ADDR = rospy.get_param('~device_address', 0x68)
    if type(ADDR) == str:
        ADDR = int(ADDR, 16)

    IMU_FRAME = rospy.get_param('~imu_frame', 'imu_link')
    print("and the addr goes to",ADDR)
    
    bus.write_byte_data(ADDR, PWR_MGMT_1,0 )
    imu_pub_spot = rospy.Publisher('spot/imu', IMUdata, queue_size=10)    

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        publish_imu_spot()
        rate.sleep()
