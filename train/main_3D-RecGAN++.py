import os
import shutil
import numpy as np
import scipy.io
import tensorflow as tf
import tools
from tqdm import tqdm
from PIL import Image


current_epoch = 0
vox_res64 = 64
# vox_rex256 = 256
batch_size = 16
GPU0 = '0'
re_train=False # continue to train: True.

#########################
config={}
config['batch_size']=batch_size
config['vox_res_x'] = vox_res64
config['vox_res_y'] = vox_res64

config['train_names'] = ['shapenet_004_sprayer']
for name in config['train_names']:
    config['X_train_'+name] = './demo_dataset/'+name+'/train_x_occ/'
    config['Y_train_'+name] = './demo_dataset/'+name+'/gt/'

config['test_names'] = ['shapenet_004_sprayer']
for name in config['test_names']:
    config['X_test_'+name]= './demo_dataset/'+name+'/test_x_occ/'
    config['Y_test_'+name]= './demo_dataset/'+name+'/gt/'
#########################

class Network:
    def __init__(self, demo_only=False):
        if demo_only:
            return  # no need to creat folders
        self.train_mod_dir = './train_mod/'
        self.train_sum_dir = './train_sum/'
        self.test_res_dir = './test_res/'
        self.test_sum_dir = './test_sum/'
        self.current_epoch = current_epoch

        print ("re_train:", re_train)
        if os.path.exists(self.test_res_dir):
            if re_train:
                print ("test_res_dir and files kept!")
            else:
                shutil.rmtree(self.test_res_dir)
                os.makedirs(self.test_res_dir)
                print ('test_res_dir: deleted and then created!')
        else:
            os.makedirs(self.test_res_dir)
            print ('test_res_dir: created!')

        if os.path.exists(self.train_mod_dir):
            if re_train:
                if os.path.exists(self.train_mod_dir + 'epoch_'+str(self.current_epoch)+'/model.cptk.data-00000-of-00001'):
                    print ('model found! will be reused!')
                else:
                    print ('model not found! error!')
                    exit()
            else:
                if os.path.exists(self.train_mod_dir):
                    yorn = str(raw_input('Find pre-existing models. Do you really want to delete all of them? [Y/N] '))
                    if yorn.startswith('y'):
                        self.current_epoch = -1
                        shutil.rmtree(self.train_mod_dir)
                        os.makedirs(self.train_mod_dir)
                        print ('train_mod_dir: deleted and then created!')
                    else:
                        print('Aborted')
                        exit()

        else:
            os.makedirs(self.train_mod_dir)
            print ('train_mod_dir: created!')

        if os.path.exists(self.train_sum_dir):
            if re_train:
                print ("train_sum_dir and files kept!")
            else:
                shutil.rmtree(self.train_sum_dir)
                os.makedirs(self.train_sum_dir)
                print ('train_sum_dir: deleted and then created!')
        else:
            os.makedirs(self.train_sum_dir)
            print ('train_sum_dir: created!')

        if os.path.exists(self.test_sum_dir):
            if re_train:
                print ("test_sum_dir and files kept!")
            else:
                shutil.rmtree(self.test_sum_dir)
                os.makedirs(self.test_sum_dir)
                print ('test_sum_dir: deleted and then created!')
        else:
            os.makedirs(self.test_sum_dir)
            print ('test_sum_dir: created!')

    def aeu(self, X):
        with tf.device('/gpu:'+GPU0):
            X = tf.reshape(X,[-1, vox_res64,vox_res64,vox_res64,1])
            c_e = [1,64,128,256,512]
            s_e = [0,1 , 1, 1, 1]
            layers_e = []
            layers_e.append(X)
            for i in range(1,5,1):
                layer = tools.Ops.conv3d(layers_e[-1],k=4,out_c=c_e[i],str=s_e[i],name='e'+str(i))
                layer = tools.Ops.maxpool3d(tools.Ops.xxlu(layer, label='lrelu'), k=2,s=2,pad='SAME')
                layers_e.append(layer)

            ### fc
            [_, d1, d2, d3, cc] = layers_e[-1].get_shape()
            d1=int(d1); d2=int(d2); d3=int(d3); cc=int(cc)
            lfc = tf.reshape(layers_e[-1],[-1, int(d1)*int(d2)*int(d3)*int(cc)])
            lfc = tools.Ops.xxlu(tools.Ops.fc(lfc, out_d=2000,name='fc1'), label='relu')

        with tf.device('/gpu:'+GPU0):
            lfc = tools.Ops.xxlu(tools.Ops.fc(lfc,out_d=d1*d2*d3*cc, name='fc2'), label='relu')
            lfc = tf.reshape(lfc, [-1, d1,d2,d3,cc])

            c_d = [0,256,128,64,16]
            s_d = [0,2,2,2,2]
            layers_d = []
            layers_d.append(lfc)
            for j in range(1,5,1):
                u_net = True
                if u_net:
                    layer = tf.concat([layers_d[-1], layers_e[-j]],axis=4)
                    layer = tools.Ops.deconv3d(layer, k=4,out_c=c_d[j], str=s_d[j],name='d'+str(len(layers_d)))
                else:
                    layer = tools.Ops.deconv3d(layers_d[-1],k=4,out_c=c_d[j],str=s_d[j],name='d'+str(len(layers_d)))

                layer = tools.Ops.xxlu(layer, label='relu')
                layers_d.append(layer)

            ###
            layer = tools.Ops.deconv3d(layers_d[-1],k=4,out_c=1,str=1,name='dlast')

            ###
            Y_sig = tf.nn.sigmoid(layer)
            Y_sig_modi = tf.maximum(Y_sig,0.01)

        return Y_sig, Y_sig_modi

    def dis(self, X, Y):
        with tf.device('/gpu:'+GPU0):
            X = tf.reshape(X,[-1, vox_res64, vox_res64, vox_res64,1])
            Y = tf.reshape(Y,[-1, vox_res64, vox_res64,vox_res64,1])
            Y = tf.concat([X, Y],axis=3)

            c_d = [1,8,16,32,64]
            s_d = [0,2,2,2,2]
            layers_d =[]
            layers_d.append(Y)
            for i in range(1,5,1):
                layer = tools.Ops.conv3d(layers_d[-1],k=4,out_c=c_d[i],str=s_d[i],name='d'+str(i))
                if i!=6:
                    layer = tools.Ops.xxlu(layer, label='lrelu')
                layers_d.append(layer)
            [_, d1, d2, d3, cc] = layers_d[-1].get_shape()
            d1 = int(d1); d2 = int(d2); d3 = int(d3); cc = int(cc)
            y = tf.reshape(layers_d[-1],[-1,d1*d2*d3*cc])
        return tf.nn.sigmoid(y)

    def build_graph(self):
        self.X = tf.placeholder(shape=[None, vox_res64, vox_res64, vox_res64, 1], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[None, vox_res64, vox_res64, vox_res64, 1], dtype=tf.float32)
        self.Mask = tf.placeholder(shape=[None, vox_res64, vox_res64, vox_res64,1], dtype=tf.float32)

        with tf.variable_scope('aeu'):
            self.Y_pred, self.Y_pred_modi = self.aeu(self.X)
        with tf.variable_scope('dis'):
            self.XY_real_pair = self.dis(self.X, self.Y*self.Mask)
        with tf.variable_scope('dis',reuse=True):
            self.XY_fake_pair = self.dis(self.X, self.Y_pred*self.Mask)

        with tf.device('/gpu:'+GPU0):
            ################################ ae loss
            Y_ = tf.reshape(self.Y, shape=[-1, vox_res64**3])
            Y_pred_modi_ = tf.reshape(self.Y_pred_modi, shape=[-1, vox_res64**3])
            Mask_ = tf.reshape(self.Mask, shape=[-1,vox_res64**3])
            w = 0.85
            self.aeu_loss = tf.reduce_mean(-tf.reduce_mean(w * Mask_ * Y_ * tf.log(Y_pred_modi_ + 1e-8), reduction_indices=[1]) -
                                       tf.reduce_mean((1 - w) * Mask_ * (1 - Y_) * tf.log(1 - Y_pred_modi_ + 1e-8), reduction_indices=[1]))

            sum_aeu_loss = tf.summary.scalar('aeu_loss', self.aeu_loss)

            ################################ wgan loss
            self.gan_g_loss = -tf.reduce_mean(self.XY_fake_pair)
            self.gan_d_loss_no_gp = tf.reduce_mean(self.XY_fake_pair) - tf.reduce_mean(self.XY_real_pair)
            sum_gan_g_loss = tf.summary.scalar('gan_g_loss', self.gan_g_loss)
            sum_gan_d_loss_no_gp = tf.summary.scalar('gan_d_loss_no_gp', self.gan_d_loss_no_gp)
            alpha = tf.random_uniform(shape=[tf.shape(self.X)[0], vox_res64 ** 3], minval=0.0, maxval=1.0)

            Y_pred_ = tf.reshape(self.Y_pred, shape=[-1, vox_res64 ** 3])
            differences_ = Y_pred_ - Y_
            interpolates = Y_ + alpha*differences_
            with tf.variable_scope('dis',reuse=True):
                XY_fake_intep = self.dis(self.X, interpolates)
            gradients = tf.gradients(XY_fake_intep, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
            self.gan_d_loss_gp = self.gan_d_loss_no_gp + 10 * gradient_penalty
            sum_gan_d_loss_gp = tf.summary.scalar('gan_d_loss_gp', self.gan_d_loss_gp)

            #################################  ae + gan loss
            gan_g_w = 20
            aeu_w = 100 - gan_g_w
            self.aeu_gan_g_loss = aeu_w*self.aeu_loss + gan_g_w*self.gan_g_loss

        with tf.device('/gpu:'+GPU0):
            aeu_var = [var for var in tf.trainable_variables() if var.name.startswith('aeu')]
            dis_var = [var for var in tf.trainable_variables() if var.name.startswith('dis')]
            self.aeu_g_optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8).\
                            minimize(self.aeu_gan_g_loss, var_list=aeu_var)
            self.dis_optim = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-8).\
                            minimize(self.gan_d_loss_gp,var_list=dis_var)

        print (tools.Ops.variable_count())
        self.sum_merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=5)#1
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.visible_device_list = GPU0

        self.sess = tf.Session(config=config)
        self.sum_writer_train = tf.summary.FileWriter(self.train_sum_dir, self.sess.graph)
        self.sum_write_test = tf.summary.FileWriter(self.test_sum_dir)

        path = self.train_mod_dir
        if os.path.isfile(path +'epoch_'+str(self.current_epoch)+ '/model.cptk.data-00000-of-00001'):
            print ('restoring saved model epoch_'+str(self.current_epoch))
            self.saver.restore(self.sess, path +'epoch_'+str(self.current_epoch)+ '/model.cptk')
        else:
            print ('initilizing model')
            self.sess.run(tf.global_variables_initializer())

        return 0

    def train(self, data):
        while self.current_epoch < 30:
            data.shuffle_X_Y_files(label='train')
            total_train_batch_num = data.total_train_batch_num
            print ('total_train_batch_num:', total_train_batch_num)
            for i in tqdm(range(total_train_batch_num)):

                #################### training
                X_train_batch, Y_train_batch, mask = data.queue_train.get()
                assert(mask.dtype=='float32')
                self.sess.run(self.dis_optim, feed_dict={self.X:X_train_batch, self.Y:Y_train_batch, self.Mask:mask})
                self.sess.run(self.aeu_g_optim, feed_dict={self.X:X_train_batch, self.Y:Y_train_batch, self.Mask:mask})

                aeu_loss_c, gan_g_loss_c, gan_d_loss_no_gp_c, gan_d_loss_gp_c, sum_train = self.sess.run(
                [self.aeu_loss, self.gan_g_loss, self.gan_d_loss_no_gp, self.gan_d_loss_gp, self.sum_merged],
                feed_dict={self.X:X_train_batch, self.Y:Y_train_batch,self.Mask:mask})

                if i%200==0:
                    self.sum_writer_train.add_summary(sum_train, self.current_epoch * total_train_batch_num + i)

                #################### testing

            print('Evaluating...')
            iou = []
            for i in tqdm(range(data.total_test_seq_batch)):
                X_test_batch, Y_test_batch, Y_masks = data.load_X_Y_voxel_grids_test_next_batch()
                Y_masks = Y_masks[:,:,:,:,0]
                Y_pred_t = self.sess.run(self.Y_pred,feed_dict={self.X:X_test_batch, self.Y:Y_test_batch})
                Y_pred_t = (Y_pred_t>0.5)[:,:,:,:,0]
                Y_test_batch = (Y_test_batch>0.5)[:,:,:,:,0]
                iou_or = np.logical_or(Y_test_batch,Y_pred_t).sum(axis=-1).sum(axis=-1).sum(axis=-1).astype(float)
                iou_and = np.logical_and(Y_test_batch,Y_pred_t).sum(axis=-1).sum(axis=-1).sum(axis=-1).astype(float)
                iou_1 = np.mean(np.divide(iou_and,iou_or))
                iou.append(iou_1)
            summary=tf.Summary()
            summary.value.add(tag='iou', simple_value = np.average(np.array(iou)))
            self.sum_write_test.add_summary(summary, self.current_epoch * total_train_batch_num + i)
            print('Evaluation iou', iou_1)

            # iou_or = (np.logical_or(Y_test_batch,Y_pred_t).astype(float)*Y_masks).sum(axis=-1).sum(axis=-1).sum(axis=-1)
            # iou_and = (np.logical_and(Y_test_batch,Y_pred_t).astype(float)*Y_masks).sum(axis=-1).sum(axis=-1).sum(axis=-1)
            # iou_2 = np.mean(np.divide(iou_and,iou_or))
            # iou_2 = np.divide(np.sum(np.logical_and(Y_test_batch,Y_pred_t)*Y_masks),np.sum(np.logical_or(Y_test_batch,Y_pred_t)*Y_masks))
            #
            # summary.value.add(tag='iou_masked', simple_value = iou_2)
            # self.sum_write_test.add_summary(summary, self.current_epoch * total_train_batch_num + i)
            self.sum_write_test.flush()

            self.current_epoch += 1
            if os.path.exists(self.train_mod_dir + 'epoch_'+str(self.current_epoch)+'/'):
                shutil.rmtree(self.train_mod_dir + 'epoch_'+str(self.current_epoch)+'/')
                print ('epoch_'+str(self.current_epoch)+'/'+': deleted and then created!')
            os.makedirs(self.train_mod_dir + 'epoch_'+str(self.current_epoch)+'/')
            self.saver.save(self.sess, save_path=self.train_mod_dir +'epoch_'+str(self.current_epoch)+ '/model.cptk')
            print ('ep:', self.current_epoch, 'model saved!')

        data.stop_queue=True

#########################
if __name__ == '__main__':
    data = tools.Data(config)
    data.daemon = True
    data.start()
    net = Network()
    net.build_graph()
    net.train(data)
