import numpy as np
# from skimage.measure import compare_ssim, compare_psnr, compare_mse, compare_nrmse
import cv2

import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops import math_ops

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', type=str, default='../scri/GeneratedData/CVUSA/pix2pix_tanpolar_L1Grd_0.0_PerGrd_1.0/')

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
    val_file = '../../../Data/CVUSA/splits/val-19zl.csv'

    id_list = []
    with open(val_file, 'r') as f:
        for line in f:
            data = line.split(',')
            pano_id = (data[0].split('/')[-1]).split('.')[0]
            id_list.append(pano_id)

    return id_list


def input_data_generator(input_dir, target_dir='../../../Data/CVUSA/streetview/targets/1/', batch_size=1):
    id_list = get_val_id_list()

    num_batches = len(id_list)//batch_size

    for i in range(num_batches + 1):

        input_list = []
        target_list = []

        img_num_per_batch = batch_size if i<num_batches else len(id_list)%batch_size

        for j in range(img_num_per_batch):
            input_list.append((cv2.imread(input_dir + id_list[i*batch_size+j] + '.png')).astype(np.float32)) # [64:, ...]

            target_img = cv2.imread(target_dir + id_list[i*batch_size+j] + '.png')
            target_list.append((cv2.resize(target_img, (512, 128))).astype(np.float32))  #
            # target_list.append((cv2.resize(target_img, (1024, 256))).astype(np.float32)[:, :256, :])  # [64:, ...]

        yield np.stack(input_list, axis=0), np.stack(target_list, axis=0)


def get_evaluation_metrics(inputs, targets):

    # inputs = tf.stack(input_list, axis=0)
    # targets = tf.stack(target_list, axis=0)

    ssim = tf.reduce_sum(tf.image.ssim(inputs, targets, max_val=255))
    psnr = tf.reduce_sum(tf.image.psnr(inputs, targets, max_val=255))
    rmse = tf.reduce_sum(RMSE(inputs, targets))
    sharpdiff = tf.reduce_sum(SharpDiff(inputs, targets))

    return ssim, psnr, rmse, sharpdiff


if __name__=="__main__":
    batch_size = 128

    # input_dir = '../../Data/CVUSA/GAN/MIP_corrceted/val/'
    # # input_dir = '../../Data/CVUSA/polarmap/19/'
    # input_list, target_list = read_input_data(input_dir)
    # print('read data done...')
    # ssim, psnr, rmse, sharpdiff = get_evaluation_metrics(tf.constant(input_list), tf.constant(target_list))
    # sess = tf.Session()
    # ssim_val, psnr_val, rmse_val, sharpdiff_val = sess.run([ssim, psnr, rmse, sharpdiff])
    #
    # print(ssim_val, psnr_val, rmse_val, sharpdiff_val)

    #input_dir = '../../ours_TF/script/GeneratedData/CVUSA/geometry_radiusPlaneMethod_column_image_image_L1Grd_100.0_PerGrd_0.0/image/'
    #input_dir = '../../ours_TF/stoa/GeneratedData/CVUSA/pix2pix/tanpolar/val/'
    # input_dir = '../../ours_Pytorch/GeneratedData/CVUSA/geoselectiongan/image/'
   
    data_generator = input_data_generator(input_dir, batch_size=batch_size)

    inputs = tf.placeholder(tf.float32, [None, 128, 512, 3], name='inputs')
    targets = tf.placeholder(tf.float32, [None, 128, 512, 3], name='targets')
    ssim, psnr, rmse, sharpdiff = get_evaluation_metrics(inputs, targets)
    # sharpdiff = SharpDiff(inputs, targets)

    ssim_sum = 0
    psnr_sum = 0
    rmse_sum = 0
    sharpdiff_sum = 0
    i = 0

    sess = tf.Session()
    for batch_inputs, batch_targets in data_generator:
        feed_dict = {inputs: batch_inputs, targets:batch_targets}
        ssim_val, psnr_val, rmse_val, sharpdiff_val = sess.run([ssim, psnr, rmse, sharpdiff], feed_dict=feed_dict)
        # sharpdiff_val = sess.run(sharpdiff, feed_dict)
        #print(i, ssim_val/batch_size)
        ssim_sum += ssim_val
        psnr_sum += psnr_val
        rmse_sum += rmse_val
        sharpdiff_sum += sharpdiff_val
        i += batch_inputs.shape[0]
    ssim_mean = ssim_sum/i
    psnr_mean = psnr_sum/i
    rmse_mean = rmse_sum/i
    sharpdiff_mean = sharpdiff_sum/i

    print('=================================================================')
    print(ssim_mean, psnr_mean, rmse_mean, sharpdiff_mean, i)
    print('=================================================================')

