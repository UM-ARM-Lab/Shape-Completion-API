import numpy as np
import os
import re
from random import shuffle
import shutil
import tensorflow as tf
import scipy.io
import scipy.misc
from scipy import ndimage
import sklearn.metrics
#import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
import copy
import math
import random
from skimage.measure import block_reduce
import threading
import sys
import binvox_rw

if sys.version_info >=(2,0):
    print (sys.version)
    import Queue as queue
if sys.version_info >=(3,0):
    print (sys.version)
    import queue
np.set_printoptions(threshold=np.nan)
class Data(threading.Thread):
    def __init__(self,config):
        super(Data,self).__init__()
        self.config = config
        self.train_batch_index = 0
        self.test_seq_index = 0

        self.batch_size = config['batch_size']
        self.vox_res_x = config['vox_res_x']
        self.vox_res_y = config['vox_res_y']
        self.train_names = config['train_names']
        self.test_names = config['test_names']

        self.queue_train = queue.Queue(3)
        self.stop_queue = False

        self.X_train_files, self.Y_train_files = self.load_X_Y_files_paths_all( self.train_names,label='train')
        self.X_test_files, self.Y_test_files = self.load_X_Y_files_paths_all(self.test_names, label='test')

        print ('X_train_files:',len(self.X_train_files))
        print ('X_test_files:',len(self.X_test_files))

        self.total_train_batch_num = int(len(self.X_train_files) // self.batch_size)
        self.total_test_seq_batch = int(len(self.X_test_files) // self.batch_size)
    '''
    @staticmethod

    def plotFromVoxels(voxels,title=''):
        if len(voxels.shape)>3:
            x_d = voxels.shape[0]
            y_d = voxels.shape[1]
            z_d = voxels.shape[2]
            v = voxels[:,:,:,0]
            v = np.reshape(v,(x_d,y_d,z_d))
        else:
            v = voxels
        x, y, z = v.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        #plt.show()
        plt.title(title)
        ax.set_xlim([0,64])
        ax.set_ylim([0,64])
        ax.set_zlim([0,64])
        from matplotlib.pyplot import show
        show(block=False)
    '''
    @staticmethod
    def vox_down_single(vox, to_res):
        from_res = vox.shape[0]
        step = int(from_res / to_res)
        vox = np.reshape(vox,[from_res,from_res,from_res])
        new_vox = block_reduce(vox,(step,step,step),func=np.max)
        new_vox = np.reshape(new_vox,[to_res,to_res,to_res,1])
        return new_vox

    @staticmethod
    def vox_down_batch(vox_bat, to_res):
        from_res = vox_bat.shape[1]
        step = int(from_res / to_res)
        new_vox_bat = []
        for b in range(vox_bat.shape[0]):
            tp = np.reshape(vox_bat[b,:,:,:,:], [from_res,from_res,from_res])
            tp = block_reduce(tp,(step,step,step),func=np.max)
            tp = np.reshape(tp,[to_res,to_res,to_res,1])
            new_vox_bat.append(tp)
        new_vox_bat = np.asarray(new_vox_bat)
        return new_vox_bat

    @staticmethod
    def voxel_grid_padding(a):
        x_d = a.shape[0]
        y_d = a.shape[1]
        z_d = a.shape[2]
        channel = a.shape[3]
        ori_vox_res = 256
        size = [ori_vox_res, ori_vox_res, ori_vox_res,channel]
        b = np.zeros(size,dtype=np.float32)

        bx_s = 0;bx_e = size[0];by_s = 0;by_e = size[1];bz_s = 0; bz_e = size[2]
        ax_s = 0;ax_e = x_d;ay_s = 0;ay_e = y_d;az_s = 0;az_e = z_d
        if x_d > size[0]:
            ax_s = int((x_d - size[0]) / 2)
            ax_e = int((x_d - size[0]) / 2) + size[0]
        else:
            bx_s = int((size[0] - x_d) / 2)
            bx_e = int((size[0] - x_d) / 2) + x_d

        if y_d > size[1]:
            ay_s = int((y_d - size[1]) / 2)
            ay_e = int((y_d - size[1]) / 2) + size[1]
        else:
            by_s = int((size[1] - y_d) / 2)
            by_e = int((size[1] - y_d) / 2) + y_d

        if z_d > size[2]:
            az_s = int((z_d - size[2]) / 2)
            az_e = int((z_d - size[2]) / 2) + size[2]
        else:
            bz_s = int((size[2] - z_d) / 2)
            bz_e = int((size[2] - z_d) / 2) + z_d
        b[bx_s:bx_e, by_s:by_e, bz_s:bz_e,:] = a[ax_s:ax_e, ay_s:ay_e, az_s:az_e, :]

        #Data.plotFromVoxels(b)
        return b

    @staticmethod
    def load_single_voxel_grid(path, out_vox_res=256):
        # with np.load(path) as da:
        #     voxel_grid = da['arr_0']
        with open(path, 'rb') as f:
            voxel_grid = binvox_rw.read_as_3d_array(f).data
            voxel_grid = np.expand_dims(voxel_grid, axis=3)
        if len(voxel_grid)<=0:
            print (" load_single_voxel_grid error: ", path)
            exit()

        # Data.plotFromVoxels(voxel_grid)
        # voxel_grid = Data.voxel_grid_padding(voxel_grid)

        # downsample
        if out_vox_res < 256:
            voxel_grid = Data.vox_down_single(voxel_grid, to_res=out_vox_res)
        return voxel_grid

    @staticmethod
    def find_nth(haystack, needle, n):
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start+len(needle))
            n -= 1
        return start

    def load_X_Y_files_paths_all(self, obj_names, label='train'):
        x_str=''
        y_str=''
        if label =='train':
            x_str='X_train_'
            y_str ='Y_train_'
        elif label == 'test':
            x_str = 'X_test_'
            y_str = 'Y_test_'
        else:
            print ("label error!!")
            exit()

        X_data_files_all = []
        Y_data_files_all = []
        for name in obj_names:
            X_folder = self.config[x_str + name]
            Y_folder = self.config[y_str + name]
            X_data_files = [X_f for X_f in sorted(os.listdir(X_folder))]
            # Y_data_files = [Y_f for Y_f in sorted(os.listdir(Y_folder))]
            Y_data_files = [x_name[0:self.find_nth(x_name,'_',5)]+'_gt.binvox' for x_name in X_data_files]
            for X_f, Y_f in zip(X_data_files, Y_data_files):
                # if X_f[0:15] != Y_f[0:15]:
                if X_f.split('occupy.binvox')[0] != Y_f.split('gt.binvox')[0]:
                    print ("index inconsistent!!")
                    exit()
                X_data_files_all.append(X_folder + X_f)
                Y_data_files_all.append(Y_folder + Y_f)
        return X_data_files_all, Y_data_files_all

    def load_X_Y_voxel_grids(self,X_data_files, Y_data_files, train = True):
        if len(X_data_files) !=self.batch_size or len(Y_data_files)!=self.batch_size:
            print ("load_X_Y_voxel_grids error:", X_data_files, Y_data_files)
            exit()

        X_voxel_grids = []
        Y_voxel_grids = []
        Y_masks = []
        index = -1
        for X_f, Y_f in zip(X_data_files, Y_data_files):
            index += 1
            X_voxel_grid = self.load_single_voxel_grid(X_f, out_vox_res=self.vox_res_x)
            X_voxel_grids.append(X_voxel_grid)

            Y_voxel_grid = self.load_single_voxel_grid(Y_f, out_vox_res=self.vox_res_y)
            Y_voxel_grids.append(Y_voxel_grid)

            assert(Y_voxel_grid.shape[3]==1)

            Y_mask = self.get_train_mask(Y_voxel_grid,3,train)

            Y_masks.append(Y_mask)

        X_voxel_grids = np.asarray(X_voxel_grids)
        Y_voxel_grids = np.asarray(Y_voxel_grids)
        Y_masks = np.asarray(Y_masks)
        # print(Y_voxel_grids.dtype,Y_mask.dtype)
        return X_voxel_grids, Y_voxel_grids, Y_masks

    def shuffle_X_Y_files(self, label='train'):
        X_new = []; Y_new = []
        if label == 'train':
            X = self.X_train_files; Y = self.Y_train_files
            self.train_batch_index = 0
            index = list(range(len(X)))
            shuffle(index)
            for i in index:
                X_new.append(X[i])
                Y_new.append(Y[i])
            self.X_train_files = X_new
            self.Y_train_files = Y_new
        else:
            print ("shuffle_X_Y_files error!\n")
            exit()

    ###################### voxel grids
    def load_X_Y_voxel_grids_train_next_batch(self):
        X_data_files = self.X_train_files[self.batch_size * self.train_batch_index:self.batch_size * (self.train_batch_index + 1)]
        Y_data_files = self.Y_train_files[self.batch_size * self.train_batch_index:self.batch_size * (self.train_batch_index + 1)]
        self.train_batch_index += 1

        X_voxel_grids, Y_voxel_grids, Y_masks = self.load_X_Y_voxel_grids(X_data_files, Y_data_files, train = True)
        return X_voxel_grids, Y_voxel_grids, Y_masks

    def load_X_Y_voxel_grids_test_next_batch(self,fix_sample=False):
        if fix_sample:
            random.seed(42)
        idx = random.sample(range(len(self.X_test_files)), self.batch_size)
        X_test_files_batch = []
        Y_test_files_batch = []
        for i in idx:
            X_test_files_batch.append(self.X_test_files[i])
            Y_test_files_batch.append(self.Y_test_files[i])

        X_test_batch, Y_test_batch, Y_masks = self.load_X_Y_voxel_grids(X_test_files_batch, Y_test_files_batch, train = False)
        return X_test_batch, Y_test_batch, Y_masks

    def get_train_mask(self, Y_, w, train = True):

        assert(Y_.shape == (64,64,64,1))
        Y_=Y_.astype(int)
        if train:
            filt = np.ones((w,w,w),dtype='float32')
            # print(Y_.shape)
            out = ndimage.convolve(Y_[:,:,:,0],filt,mode='constant',cval=0)
            mask = np.ones(Y_.shape,dtype='float32')
            # mask[:,:,:,0] += 0.5*np.logical_and(out>0,out<w**3)

            #mask[:,:,:,0]-=(out==w**3).astype(float)

            # mask = np.logical_and(out>0,out<w**3)
            # print(mask[:,:,32,0])
            return mask
        else: #test
            filt = np.ones((w,w,w),dtype='float32')
            out = ndimage.convolve(Y_[:,:,:,0],filt,mode='constant',cval=0)

            mask = np.ones(Y_.shape,dtype='float32')
            mask[:,:,:,0]-=(out==w**3).astype(float)
            return mask
    def run(self):
        while not self.stop_queue:
            ## train
            if not self.queue_train.full():
                if self.train_batch_index>=self.total_train_batch_num:
                    self.shuffle_X_Y_files(label='train')
                    print ('shuffle')
                X_b, Y_b, mask = self.load_X_Y_voxel_grids_train_next_batch()


                self.queue_train.put((X_b, Y_b, mask))

class Ops:

    @staticmethod
    def lrelu(x, leak=0.2):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    @staticmethod
    def relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def xxlu(x,label,name=None):
        if label =='relu':
            return  Ops.relu(x)
        if label =='lrelu':
            return  Ops.lrelu(x,leak=0.2)

    @staticmethod
    def variable_sum(var, name):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def variable_count():
        total_para = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        return total_para

    @staticmethod
    def fc(x, out_d, name):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_d = x.get_shape()[1]
        w = tf.get_variable(name + '_w', [in_d, out_d], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_d], initializer=zero_init)
        y = tf.nn.bias_add(tf.matmul(x, w), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def maxpool3d(x,k,s,pad='SAME'):
        ker =[1,k,k,k,1]
        str =[1,s,s,s,1]
        y = tf.nn.max_pool3d(x,ksize=ker,strides=str,padding=pad)
        return y

    @staticmethod
    def conv3d(x, k, out_c, str, name,pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        in_c = x.get_shape()[4]
        w = tf.get_variable(name + '_w', [k, k, k, in_c, out_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
        stride = [1, str, str, str, 1]
        y = tf.nn.bias_add(tf.nn.conv3d(x, w, stride, pad), b)
        Ops.variable_sum(w, name)
        return y

    @staticmethod
    def deconv3d(x, k, out_c, str, name,pad='SAME'):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.zeros_initializer()
        [_, in_d1, in_d2, in_d3, in_c] = x.get_shape()
        in_d1 = int(in_d1); in_d2 = int(in_d2); in_d3 = int(in_d3); in_c = int(in_c)
        bat = tf.shape(x)[0]
        w = tf.get_variable(name + '_w', [k, k, k, out_c, in_c], initializer=xavier_init)
        b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
        out_shape = [bat, in_d1 * str, in_d2 * str, in_d3 * str, out_c]
        stride = [1, str, str, str, 1]
        y = tf.nn.conv3d_transpose(x, w, output_shape=out_shape, strides=stride, padding=pad)
        y = tf.nn.bias_add(y, b)
        Ops.variable_sum(w, name)
        return y
