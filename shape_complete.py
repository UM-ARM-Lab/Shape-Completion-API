import os
import numpy as np
import tensorflow as tf

# config
GPU0 = '/gpu:0'
model_path = './train_mod/'
RESOLUTION = 64

class Shape_complete():
    def __init__(self, verbose = False):
        '''
        Constructor of the Shape_complete class. Load the model from 'model_path'.
        INPUT: verbose: print messages for debug
        '''
        if not verbose:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
            tf.logging.set_verbosity(tf.logging.FATAL)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.import_meta_graph( model_path + 'model.cptk.meta', clear_devices=True)
        self.saver.restore(self.sess, model_path+'model.cptk')
        if verbose:
            print ('model restored!')

        self.X_occ = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
        self.X_non = tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
        self.Y_pred = tf.get_default_graph().get_tensor_by_name('aeu/Sigmoid:0')
    

    def complete(self ,occ, non, verbose = False):  
        '''
        Complete the 3d shape according to the occupied grids and non-occupied grids
        INPUT: occ: DATATYPE: bool. SHAPE: (64,64,64) for a single occupied grids OR (batch_size,64, 64, 64) for a batch of occupied grids
        INPUT: non: DATATYPE: bool. SHAPE: (64,64,64), dtype: bool. for a single occupied grids OR (batch_size,64, 64, 64) for a batch of occupied grids(64,64,64) for a single occupied grids OR (batch_size,64, 64, 64) for a batch of occupied grids: non: (64,64,64) for a single non-occupied grids OR (batch_size,64, 64, 64) for a batch of non-occupied grids
        INPUT: verbose: DATATYPE: bool. give some messages for debug
        OUTPUT: completed shape: DATATYPE: bool. SHAPE (64,64,64) for a single occupied grids OR (batch_size,64, 64, 64) for a batch of occupied grids
        '''
        if not occ.shape == non.shape:
            raise ValueError('Error! Wrong dimensions')
        if occ.ndim == 3 and occ.ndim == 3:
            if verbose:
                out_dim = 3
                print('Get input as single voxel')
            assert(occ.shape == (RESOLUTION,RESOLUTION,RESOLUTION))
            assert(non.shape == (RESOLUTION,RESOLUTION,RESOLUTION))
            occ = np.expand_dims(occ,0)
            non = np.expand_dims(non,0)
            occ = np.expand_dims(occ,4)
            non = np.expand_dims(non,4)
        elif occ.ndim == 4 and non.ndim == 4:
            if verbose:
                out_dim = 4
                print('Get input as batches. Batch size: {}'.format(occ.shape[0]))
            assert(occ.shape[-3:] == (RESOLUTION,RESOLUTION,RESOLUTION))
            assert(non.shape[-3:] == (RESOLUTION,RESOLUTION,RESOLUTION))
            occ = np.expand_dims(occ,4)
            non = np.expand_dims(non,4)
        else:
            raise ValueError('Error! Wrong dimensions')
        y_pred = self.sess.run(self.Y_pred, feed_dict={self.X_occ: occ, self.X_non: non})

        # Thresholding. Threshold sets to be 0.5
        th = 0.5
        y_pred[y_pred >= th] = 1
        y_pred[y_pred < th] = 0

        if out_dim == 4:
            return y_pred[:,:,:,0]
        elif out_dim == 3:
            return y_pred[0,:,:,:,0]
        else:
            raise ValueError('Internal error')
    def __del__(self):
        self.sess.close()



###DEMO###

def demo():
    '''
    demo on how to use this class.
    '''
    import binvox_rw
    # Constructor
    sc = Shape_complete(verbose=True)
    
    # Read demo binvox as (64*64*64) array
    with open('demo/_5_5_5_3_occupy.binvox', 'rb') as f:
        occ = binvox_rw.read_as_3d_array(f).data
    with open('demo/_5_5_5_3_non_occupy.binvox', 'rb') as f:
        non = binvox_rw.read_as_3d_array(f).data

    # Complete shape
    out = sc.complete(occ=occ,non=non,verbose=True)

    # Save to file for demo
    vox = binvox_rw.Voxels(out, [64,64,64], [0,0,0], 1, 'xyz')
    with open('demo/output.binvox','wb') as f:
        vox.write(f)
        print('Output saved to demo/output.binvox.')
        print('Please use ./binvox demo/output.binvox to visualize the result.')

if __name__ == '__main__':
    demo()
