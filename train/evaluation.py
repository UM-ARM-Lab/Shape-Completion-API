import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy import ndimage
import shutil
import sys

import binvox_rw
import chamfer_distance_api
import shape_complete
import time
import tools

data_path = './demo_dataset/'
result_path = './evaluation_result/'
vox_res64 = 64
batch_size = 22
GPU0 = '0'
SAVE_RESULT = True

test_names = [
    'shapenet_004_sprayer'
]


def get_train_mask(Y_, w, train = True):
    assert(Y_.shape == (64,64,64,1))
    Y_=Y_.astype(int)
    if train:
        filt = np.ones((w,w,w),dtype='float32')
        # print(Y_.shape)
        out = ndimage.convolve(Y_[:,:,:,0],filt,mode='constant',cval=0)
        mask = np.ones(Y_.shape,dtype='float32')
        # mask[:,:,:,0] += 0.5*np.logical_and(out>0,out<w**3)

        mask[:,:,:,0]-=(out==w**3).astype(float)

        # mask = np.logical_and(out>0,out<w**3)
        # print(mask[:,:,32,0])
        return mask
    else: #test
        filt = np.ones((w,w,w),dtype='float32')
        out = ndimage.convolve(Y_[:,:,:,0],filt,mode='constant',cval=0)

        mask = np.ones(Y_.shape,dtype='float32')
        mask[:,:,:,0]-=(out==w**3).astype(float)
        return mask


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start
def load_X_Y_voxel_grids(X_data_files_occ, X_data_files_non, Y_data_files, Known_files, batch_size,
                         vox_res_x=64, vox_res_y=64):
    if len(X_data_files_occ) != batch_size or len(X_data_files_non) != batch_size or len(Y_data_files) != batch_size:
        print ("load_X_Y_voxel_grids error:", X_data_files_occ, Y_data_files)
        exit()

    X_voxel_grids_occ = []
    X_voxel_grids_non = []
    Y_voxel_grids = []
    Y_masks = []
    Known_grids = []
    index = -1
    for X_f_occ, X_f_non, Y_f, k_f in zip(X_data_files_occ, X_data_files_non, Y_data_files, Known_files):
        index += 1
        X_voxel_grid_occ = tools.Data.load_single_voxel_grid(X_f_occ, out_vox_res=vox_res_x)
        X_voxel_grids_occ.append(X_voxel_grid_occ)

        X_voxel_grid_non = tools.Data.load_single_voxel_grid(X_f_non, out_vox_res=vox_res_x)
        X_voxel_grids_non.append(X_voxel_grid_non)

        Y_voxel_grid = tools.Data.load_single_voxel_grid(Y_f, out_vox_res=vox_res_y)
        Y_voxel_grids.append(Y_voxel_grid)

        Known_grid = tools.Data.load_single_voxel_grid(k_f, out_vox_res=vox_res_x)
        Known_grids.append(Known_grid)

        Y_mask = get_train_mask(Y_voxel_grid,3,False)       
        Y_masks.append(Y_mask)

    X_voxel_grids_occ = np.asarray(X_voxel_grids_occ)
    X_voxel_grids_non = np.asarray(X_voxel_grids_non)
    Y_voxel_grids = np.asarray(Y_voxel_grids)
    Known_grids = np.asarray(Known_grids)
    Y_masks = np.asarray(Y_masks)
    return X_voxel_grids_occ, X_voxel_grids_non, Y_voxel_grids, Known_grids, Y_masks

def evaluate():
    TIMEUSED = 0
    TOTALPOINTS = 0
    for test_name in test_names:
        sc = shape_complete.Shape_complete(verbose=True)
        if SAVE_RESULT:
            if os.path.exists(result_path+test_name):
                shutil.rmtree(result_path+test_name)
            os.mkdir(result_path+test_name)
            verbose_file_iou = open(result_path+"detail_"+test_name+"_iou.txt","w+")
            verbose_file_cd = open(result_path+"detail_"+test_name+"_cd.txt","w+")

        test_files_x_occ = sorted(os.listdir(data_path+test_name+'/test_x_occ'))
        test_files_x_non = [x_name[0:find_nth(x_name,'_',5)]+'_non_occupy.binvox' for x_name in test_files_x_occ]
        test_files_y = [x_name[0:find_nth(x_name,'_',5)]+'_gt.binvox' for x_name in test_files_x_occ]
        test_files_known = [x_name[0:find_nth(x_name,'_',5)]+'_mask.binvox' for x_name in test_files_x_occ]
        

        batches_x_occ = [test_files_x_occ[i:i + batch_size] for i in xrange(0, len(test_files_x_occ), batch_size)]
        batches_x_non = [test_files_x_non[i:i + batch_size] for i in xrange(0, len(test_files_x_non), batch_size)]
        batches_y = [test_files_y[i:i + batch_size] for i in xrange(0, len(test_files_y), batch_size)]
        batches_known = [test_files_known[i:i + batch_size] for i in xrange(0, len(test_files_known), batch_size)]

        batches_x_occ_name = np.copy(batches_x_occ)
        batches_x_non_name = np.copy(batches_x_non)
        batches_y_name = np.copy(batches_y)
        batches_known_name = np.copy(batches_known)
        for i in range(len(batches_x_occ)):
            for j in range(len(batches_x_occ[0])):
                batches_x_occ[i][j] = data_path+test_name+'/test_x_occ/'+batches_x_occ[i][j]
                batches_x_non[i][j] = data_path+test_name+'/non_occupy/'+batches_x_non[i][j]
                batches_y[i][j] = data_path+test_name+'/gt/'+batches_y[i][j]
                batches_known[i][j] = data_path+test_name+'/mask/'+batches_known[i][j]
        out_total = []
        for batch_i in tqdm(range(len(batches_x_occ))):
            x_sample_occ, x_sample_non, y_true, y_known, y_masks = load_X_Y_voxel_grids(batches_x_occ[batch_i],batches_x_non[batch_i],
                                    batches_y[batch_i],batches_known[batch_i], len(batches_x_occ[0]))
            # Complete shape
            out_total.append(sc.complete(occ=x_sample_occ[:,:,:,:,0],non=x_sample_non[:,:,:,:,0],verbose=True))
        #Evaluation
        print('Begin Evaluation')
        cd_calculator = chamfer_distance_api.Chamfer_distance(verbose = True)
        for batch_i in tqdm(range(len(batches_x_occ))):
            x_sample_occ, x_sample_non, y_true, y_known, y_masks = load_X_Y_voxel_grids(batches_x_occ[batch_i],batches_x_non[batch_i],
                                    batches_y[batch_i],batches_known[batch_i], len(batches_x_occ[0]))
            out = out_total[batch_i]
            for ot, gt, name in zip(out, y_true[:,:,:,:,0], batches_x_occ_name[batch_i]):
                out_sparse = binvox_rw.dense_to_sparse(ot)
                gt_sparse = binvox_rw.dense_to_sparse(gt)
                cd = cd_calculator.get_chamfer_distance(out_sparse.T, gt_sparse.T, verbose=True)
                # print('cd: {}'.format(cd))

                intersect = np.sum(np.logical_and(ot,gt))
                union = np.sum(np.logical_or(ot, gt))
                iou = intersect*1.0/union
                # print('iou: {}'.format(iou))

                if SAVE_RESULT:
                    verbose_file_iou.write('{0}: {1}\n'.format(name, iou))                        
                    verbose_file_cd.write('{0}: {1}\n'.format(name, cd)) 
                    vox = binvox_rw.Voxels(ot, [64,64,64], [0,0,0], 1, 'xyz')
                    with open(result_path+test_name+'/'+name,'wb') as f:
                        vox.write(f)
        if SAVE_RESULT:
            verbose_file_iou.close()
            verbose_file_cd.close()
        
                
               



if __name__ == '__main__':
    evaluate()


