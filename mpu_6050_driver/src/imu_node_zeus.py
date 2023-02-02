#!/usr/bin/env python2.7

import time
import smbus
import struct
import rospy
import numpy as np
from sensor_msgs.msg import Temperature, Imu
from tf.transformations import quaternion_about_axis
from mpu_6050_driver.registers import PWR_MGMT_1, ACCEL_XOUT_H, ACCEL_YOUT_H, ACCEL_ZOUT_H, TEMP_H,\
    GYRO_XOUT_H, GYRO_YOUT_H, GYRO_ZOUT_H
import sys
sys.path.append('../../')
from mini_ros.msg import IMUdata
from imu_orange_mod import IMU

ADDR = None
bus = None
IMU_FRAME = None

DISCON = False
# read_word and read_word_2c from http://blog.bitify.co.uk/2013/11/reading-data-from-mpu-6050-on-raspberry.html

def get_imu_spot():
    magn_vals = (1, 1, 1)
    accel_x = read_word_2c(ACCEL_XOUT_H)  / 16384.0
    accel_y = read_word_2c(ACCEL_YOUT_H) / 16384.0
    accel_z = read_word_2c(ACCEL_ZOUT_H)  / 16384.0
    acc_vals = (accel_x, accel_y,  accel_z)

    gyro_x = read_word_2c(GYRO_XOUT_H) / 131.0
    gyro_y = read_word_2c(GYRO_YOUT_H) / 131.0
    gyro_z = read_word_2c(GYRO_ZOUT_H)/ 131.0
    gyro_vals = (gyro_x, gyro_y, gyro_z)

    return acc_vals, magn_vals, gyro_vals

def read_word(adr):
    while True:
        try:
            if DISCON:
                bus = smbus.SMBus(1)
                ADDR = rospy.get_param('~device_address', 0x68)
                if type(ADDR) == str:
                    ADDR = int(ADDR, 16)
                bus.write_byte_data(ADDR, PWR_MGMT_1,0 )
                DISCON = False

            high = bus.read_byte_data(ADDR, adr)
            low = bus.read_byte_data(ADDR, adr+1)
            val = (high << 8) + low
            return val
        except:
            DISCON = True

def read_word_2c(adr):
    val = read_word(adr)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val


def publish_imu_spot():
    # Read the acceleration vals
    accel_x = read_word_2c(ACCEL_XOUT_H) / 16384.0
    accel_y = read_word_2c(ACCEL_YOUT_H) / 16384.0
    accel_z = read_word_2c(ACCEL_ZOUT_H) / 16384.0
    
    # Calculate a quaternion representing the orientation
    accel = accel_x, accel_y, accel_z
    ref = np.array([0, 0, 1])
    acceln = accel / np.linalg.norm(accel)
    axis = np.cross(acceln, ref)
    angle = np.arccos(np.dot(acceln, ref))
    orientation = quaternion_about_axis(angle, axis)
    orx, ory, orz, orw = orientation

    # Read the gyro vals
    gyro_x = read_word_2c(GYRO_XOUT_H) / 131.0
    gyro_y = read_word_2c(GYRO_YOUT_H) / 131.0
    gyro_z = read_word_2c(GYRO_ZOUT_H) / 131.0
    
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


def publish_imu_org():
    imu_dat = IMUdata()

    imu_class.filter_rpy()
    imu_dat.roll = np.radians(imu_class.true_roll)
    imu_dat.pitch = np.radians(imu_class.true_pitch)
    imu_dat.acc_x = imu_class.imu_data[3]
    imu_dat.acc_y = imu_class.imu_data[4]
    imu_dat.acc_z = imu_class.imu_data[5]
    imu_dat.gyro_x = imu_class.imu_data[0]
    imu_dat.gyro_y = imu_class.imu_data[1]
    imu_dat.gyro_z = imu_class.imu_data[2]
    # imu_read = True

    imu_pub_spot.publish(imu_dat)

temp_pub = None
imu_pub = None
imu_pub_spot = None

imu_class = IMU(rp_flip= True, r_neg=False, p_neg=False,  sensor_func = get_imu_spot)
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

        # publish_imu_spot()
        publish_imu_org()

        rate.sleep()
