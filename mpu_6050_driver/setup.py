#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['mpu_6050_driver'],
    # scripts=['src/imu_node.py'],
    # package_dir={'': 'src'},
    package_dir={'../mpu_6050_driver/src': ''},
    install_requires=['rospkg'],
    )

setup(**d)
