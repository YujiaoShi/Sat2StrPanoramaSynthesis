import numpy as np
# from skimage.measure import compare_ssim, compare_psnr, compare_mse, compare_nrmse
import cv2
import scipy.io as scio

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops import math_ops

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', type=str, default='../script3/GeneratedData/CVACT/geometry_radiusPlaneMethod_1_column_image_image_L1Grd_0.0_PerGrd_1.0_skip_0/image/')


opt = parser.parse_args()

input_dir = opt.dir

def safe_divide(numerator, denominator, name='safe_divide'):

    return tf.where(math_ops.greater(denominator, 0),
                    math_ops.divide(numerator, denominator),
                    tf.zeros_like(numerator), name=name)


def RMSE(input, target):
    return tf.sqrt(tf.reduce_mean((input - target)**2, axis=(1, 2, 3)))


def SharpDiff(inputs, targets):
    '''
    :param inputs: shape = [batch, height, width, channel]
    :param target: shape = [batch, height, width, channel]
    :param eps:
    :return:
    '''
    s = inputs.get_shape().as_list()
    gradx_in, grady_in = tf.image.image_gradients(inputs)
    gradx_ta, grady_ta = tf.image.image_gradients(targets)
    diff_gradients = tf.abs(gradx_in - gradx_ta)[:, 1: s[1]-1, 1: s[2]-1, :] + tf.abs(grady_in - grady_ta)[:, 1: s[1]-1, 1: s[2]-1, :]
    prediction_error = 64* tf.reduce_mean(diff_gradients, axis=[1, 2, 3])

    sharpdiff = 10 * tf.log(255.*255./prediction_error)/tf.log(10.)

    return sharpdiff


def get_val_id_list():
    allDataList = '../../../Data/CVACT/ACT_data.mat'
    img_root = '../../../Data/CVACT/'

    exist_aer_list = os.listdir(img_root + 'satview_correct')
    exist_grd_list = os.listdir(img_root + 'streetview')

    __cur_allid = 0  # for training

    # load the mat
    anuData = scio.loadmat(allDataList)

    data_list = []
    for i in range(0, len(anuData['panoIds'])):
        grd_id_align = anuData['panoIds'][i] + '_grdView.png'
        sat_id_ori = anuData['panoIds'][i] + '_satView_polish.png'
        data_list.append([grd_id_align, sat_id_ori, anuData['panoIds'][i]])

    val_inds = anuData['valSet']['valInd'][0][0] - 1
    valNum = len(val_inds)
    valList = []
    for k in range(valNum):
        valList.append(data_list[val_inds[k][0]])

    pano_list = [item[0] for item in valList if item[0] in exist_grd_list and item[1] in exist_aer_list]

    return pano_list


def input_data_generator(input_dir, target_dir='../../../Data/CVACT/targets/street/', batch_size=128):
    id_list = get_val_id_list()

    num_batches = len(id_list)//batch_size

    for i in range(num_batches + 1):

        input_list = []
        target_list = []

        img_num_per_batch = batch_size if i<num_batches else len(id_list)%batch_size

        for j in range(img_num_per_batch):#.replace('_grdView', '_satView_polish')
            # input_img = cv2.imread(input_dir + id_list[i*batch_size+j].replace('_grdView', '_satView_polish'))
            input_img = cv2.imread(input_dir + id_list[i * batch_size + j])
            input_img = cv2.resize(input_img, (512, 128))
            input_list.append((input_img).astype(np.float32))

            target_img = cv2.imread(target_dir + id_list[i*batch_size+j])
            target_img = cv2.resize(target_img, (512, 128))
            target_list.append((target_img).astype(np.float32))

        # for img_id in id_list:
        #     input_list.append((cv2.imread(input_dir + img_id + '.png')).astype(np.float32))
        #     target_list.append((cv2.imread(target_dir + img_id + '.png')).astype(np.float32))

        yield np.stack(input_list, axis=0), np.stack(target_list, axis=0)


def get_evaluation_metrics(inputs, targets):

    # inputs = tf.stack(input_list, axis=0)
    # targets = tf.stack(target_list, axis=0)

    ssim = tf.reduce_mean(tf.image.ssim(inputs, targets, max_val=255))
    psnr = tf.reduce_mean(tf.image.psnr(inputs, targets, max_val=255))
    rmse = tf.reduce_mean(RMSE(inputs, targets))
    sharpdiff = tf.reduce_mean(SharpDiff(inputs, targets))

    return ssim, psnr, rmse, sharpdiff


if __name__=="__main__":

    
    #input_dir = '../../ours_TF/script/GeneratedData/CVACT/geometry_radiusPlaneMethod_point_volume_image_L1Grd_0.0_PerGrd_1.0/image/'
    #input_dir = '../../ours_Pytorch/GeneratedData/CVACT/geoselectiongan/image/'
    #input_dir = '../../ours_TF/stoa/GeneratedData/CVACThalf/XforkPer/val/'
    #input_dir = '../../ours_TF/script/GeneratedData/CVACT/geometry_radiusPlaneMethod_column_image_image_L1Grd_0.0_PerGrd_1.0/image/'
    data_generator = input_data_generator(input_dir)

    inputs = tf.placeholder(tf.float32, [None, 128, 512, 3], name='inputs')
    targets = tf.placeholder(tf.float32, [None, 128, 512, 3], name='targets')
    ssim, psnr, rmse, sharpdiff = get_evaluation_metrics(inputs, targets)

    ssim_sum = 0
    psnr_sum = 0
    rmse_sum = 0
    sharpdiff_sum = 0
    i = 0

    sess = tf.Session()
    for batch_inputs, batch_targets in data_generator:
        feed_dict = {inputs: batch_inputs, targets:batch_targets}
        ssim_val, psnr_val, rmse_val, sharpdiff_val = sess.run([ssim, psnr, rmse, sharpdiff], feed_dict=feed_dict)
        print(i, ssim_val)
        ssim_sum += ssim_val
        psnr_sum += psnr_val
        rmse_sum += rmse_val
        sharpdiff_sum += sharpdiff_val
        i += 1
    ssim_mean = ssim_sum/i
    psnr_mean = psnr_sum/i
    rmse_mean = rmse_sum/i
    sharpdiff_mean = sharpdiff_sum/i

    print('=================================================================')
    print(ssim_mean, psnr_mean, rmse_mean, sharpdiff_mean)
    print('=================================================================')


