# Shape-completion

## Introduction
- `shape_completion.py`: provides a class that uses the trained model to do shape completion.
- `train_mod/`: contains the model.
- `demo/`: contains sample input in the shape_completion demo. 
- `binvox_rw/`: Python module to read and write .binvox files. [dimatura/binvox-rw-py](https://github.com/dimatura/binvox-rw-py)
- `viewvox`: Reads a 3D voxel file as produced by binvox or thinvox and shows it in a window. [3D voxel model viewer](http://www.patrickmin.com/viewvox/)

- The current model is trained on a subset of objects in ycb dataset and shapenet:

![](https://github.com/UM-ARM-Lab/Shape-completion/blob/master/train_mod/training_set.png)



## Prerequisites
The codes are tested on
- [`CUDA`](https://developer.nvidia.com/cuda-toolkit) 9.0 
- [`Python`](https://www.python.org) 2.7.12
- [`TensorFlow`](https://github.com/tensorflow/tensorflow) 1.7.0
- [`numpy`](http://www.numpy.org/) 1.14.2


