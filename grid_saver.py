#!/usr/bin/env python

from std_msgs.msg import ByteMultiArray, MultiArrayDimension
from rospy.numpy_msg import numpy_msg

import rospy

import numpy as np
import time
from datetime import datetime
import binvox_rw

import roslib
PKG = 'mps_voxels'
roslib.load_manifest(PKG)

DIM = 64


def callback(msg):
    arr = np.reshape(msg.data, tuple(d.size for d in msg.layout.dim))
    print rospy.get_name(), "I heard %s"%str(arr)
    occ = arr > 0
    # Save to file for demo
    vox = binvox_rw.Voxels(occ, [64, 64, 64], [0, 0, 0], 1, 'xyz')
    timestr = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    with open('voxels_' + timestr + '.binvox', 'wb') as f:
        vox.write(f)
        print('Output saved to "voxels_' + timestr + '.binvox".')
    

def listener():
    """
    Write ROS data to file
    """

    rospy.init_node('voxel_grid_saver')
    rospy.Subscriber("local_occupancy_predicted", numpy_msg(ByteMultiArray), callback)

    rospy.spin()


if __name__ == '__main__':
    listener()
