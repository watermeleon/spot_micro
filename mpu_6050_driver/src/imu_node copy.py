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


ADDR = None
bus = None
IMU_FRAME = None

DISCON = False
# read_word and read_word_2c from http://blog.bitify.co.uk/2013/11/reading-data-from-mpu-6050-on-raspberry.html
def read_word(adr):
    while True:
    # if bus.read_byte(ADDR):
        try:
            if DISCON:
                # print("trying again")
                bus = smbus.SMBus(1)
                ADDR = rospy.get_param('~device_address', 0x68)
                if type(ADDR) == str:
                    ADDR = int(ADDR, 16)
                bus.write_byte_data(ADDR, PWR_MGMT_1,0 )
                DISCON = False

            high = bus.read_byte_data(ADDR, adr)
            low = bus.read_byte_data(ADDR, adr+1)
            val = (high << 8) + low
            # print("succeed")
            return val
        except:
            # print("stuck heeeerrr")
            DISCON = True
            # pass

def read_word_2c(adr):
    val = read_word(adr)
    # try:
    #     val = read_word(adr)
    # except:
    #     return read_word_2c(adr)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

# def publish_temp(timer_event):
#     temp_msg = Temperature()
#     temp_msg.header.frame_id = IMU_FRAME
#     temp_msg.temperature = read_word_2c(TEMP_H)/340.0 + 36.53
#     temp_msg.header.stamp = rospy.Time.now()
    # temp_pub.publish(temp_msg)


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

# def publish_imu(timer_event):
#     imu_msg = Imu()
#     imu_msg.header.frame_id = IMU_FRAME

#     # Read the acceleration vals
#     accel_x = read_word_2c(ACCEL_XOUT_H) / 16384.0
#     accel_y = read_word_2c(ACCEL_YOUT_H) / 16384.0
#     accel_z = read_word_2c(ACCEL_ZOUT_H) / 16384.0
    
#     # Calculate a quaternion representing the orientation
#     accel = accel_x, accel_y, accel_z
#     ref = np.array([0, 0, 1])
#     acceln = accel / np.linalg.norm(accel)
#     axis = np.cross(acceln, ref)
#     angle = np.arccos(np.dot(acceln, ref))
#     orientation = quaternion_about_axis(angle, axis)

#     # Read the gyro vals
#     gyro_x = read_word_2c(GYRO_XOUT_H) / 131.0
#     gyro_y = read_word_2c(GYRO_YOUT_H) / 131.0
#     gyro_z = read_word_2c(GYRO_ZOUT_H) / 131.0
    
#     # Load up the IMU message
#     o = imu_msg.orientation
#     o.x, o.y, o.z, o.w = orientation

#     imu_msg.linear_acceleration.x = accel_x
#     imu_msg.linear_acceleration.y = accel_y
#     imu_msg.linear_acceleration.z = accel_z

#     imu_msg.angular_velocity.x = gyro_x
#     imu_msg.angular_velocity.y = gyro_y
#     imu_msg.angular_velocity.z = gyro_z

#     imu_msg.header.stamp = rospy.Time.now()

#     imu_pub.publish(imu_msg)


temp_pub = None
imu_pub = None
imu_pub_spot = None

if __name__ == '__main__':
    rospy.init_node('imu_node')

    # bus = smbus.SMBus(rospy.get_param('~bus', 1))
    print(" params says its:", rospy.get_param('~bus', 1))
    bus = smbus.SMBus(1)

    ADDR = rospy.get_param('~device_address', 0x68)
    if type(ADDR) == str:
        ADDR = int(ADDR, 16)

    IMU_FRAME = rospy.get_param('~imu_frame', 'imu_link')
    print("and the addr goes to",ADDR)
    
    bus.write_byte_data(ADDR, PWR_MGMT_1,0 )

    # temp_pub = rospy.Publisher('temperature', Temperature,  queue_size=10)
    imu_pub_spot = rospy.Publisher('spot/imu', IMUdata, queue_size=10)

    # temp_timer = rospy.Timer(rospy.Duration(10), publish_temp)
    # imu_timer2 = rospy.Timer(rospy.Duration(0.04), publish_imu_spot)

    # rospy.spin()

    
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        # This is called continuously. Has timeout functionality too
        # mini_commander.move_legs()
        publish_imu_spot()
        rate.sleep()
