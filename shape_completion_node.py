#!/usr/bin/env python

from shape_complete import Shape_complete
from std_msgs.msg import ByteMultiArray, MultiArrayDimension
from rospy.numpy_msg import numpy_msg

import rospy

import numpy as np

DIM = 64

def callback(msg, args):
    sc = args[0]
    pub = args[1]

    arr = np.reshape(msg.data, tuple(d.size for d in msg.layout.dim))
    print rospy.get_name(), "I heard %s"%str(arr)

    occ = arr > 0
    non = arr < 0

    out = sc.complete(occ=occ,non=non,verbose=False)

    out_msg = ByteMultiArray()
    out_msg.data = out.flatten().tolist()
    out_msg.layout.dim.append(MultiArrayDimension(label='x', size=DIM, stride=DIM*DIM*DIM))
    out_msg.layout.dim.append(MultiArrayDimension(label='y', size=DIM, stride=DIM*DIM))
    out_msg.layout.dim.append(MultiArrayDimension(label='z', size=DIM, stride=DIM))
    pub.publish(out_msg)

def listener():
    rospy.init_node('shape_completer')

    sc = Shape_complete(verbose=True)

    pub = rospy.Publisher('local_occupancy_predicted', numpy_msg(ByteMultiArray), queue_size=10)
    rospy.Subscriber("local_occupancy", numpy_msg(ByteMultiArray), callback, (sc, pub))
    rospy.spin()

if __name__ == '__main__':
    listener()