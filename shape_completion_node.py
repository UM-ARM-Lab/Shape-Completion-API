#!/usr/bin/env python

from shape_complete import Shape_complete
from std_msgs.msg import ByteMultiArray, MultiArrayDimension
from mps_voxels.srv import CompleteShape, CompleteShapeRequest, CompleteShapeResponse
from rospy.numpy_msg import numpy_msg

import rospy

import numpy as np

DIM = 64


def service_callback(req, args):
    sc = args

    arr = np.reshape(req.observation.data, tuple(d.size for d in req.observation.layout.dim))

    occ = arr > 0
    non = arr < 0

    out = sc.complete(occ=occ, non=non, verbose=False)

    resp = CompleteShapeResponse()
    resp.hypothesis.data = out.flatten().tolist()
    resp.hypothesis.layout.dim.append(MultiArrayDimension(label='x', size=DIM, stride=DIM*DIM*DIM))
    resp.hypothesis.layout.dim.append(MultiArrayDimension(label='y', size=DIM, stride=DIM*DIM))
    resp.hypothesis.layout.dim.append(MultiArrayDimension(label='z', size=DIM, stride=DIM))

    return resp


def callback(msg, args):
    sc = args[0]
    pub = args[1]

    arr = np.reshape(msg.data, tuple(d.size for d in msg.layout.dim))

    occ = arr > 0
    non = arr < 0

    out = sc.complete(occ=occ, non=non, verbose=False)

    out_msg = ByteMultiArray()
    out_msg.data = out.flatten().tolist()
    out_msg.layout.dim.append(MultiArrayDimension(label='x', size=DIM, stride=DIM*DIM*DIM))
    out_msg.layout.dim.append(MultiArrayDimension(label='y', size=DIM, stride=DIM*DIM))
    out_msg.layout.dim.append(MultiArrayDimension(label='z', size=DIM, stride=DIM))
    pub.publish(out_msg)


def listener():
    rospy.init_node('shape_completer')

    sc = Shape_complete(verbose=True)

    server = rospy.Service('complete_shape', CompleteShape, lambda msg: service_callback(msg, sc))

    pub = rospy.Publisher('local_occupancy_predicted', numpy_msg(ByteMultiArray), queue_size=10)
    rospy.Subscriber("local_occupancy", numpy_msg(ByteMultiArray), callback, (sc, pub))
    rospy.spin()


if __name__ == '__main__':
    listener()
