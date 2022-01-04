import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import tensorflow as tf
from VGG.vgg import build_vgg19
import cv2
import numpy as np

x = np.arange(0, 224*224*3).reshape((1, 224, 224, 3))/(224*224*3)*255

x1 = tf.constant(x)

vgg_model_file = './imagenet-vgg-verydeep-19.mat'

net = build_vgg19(x1, vgg_model_file)

a = 1

#
# img1 = cv2.resize(cv2.imread('0000011.jpg'), (256, 256)).astype(np.float32)[np.newaxis,...]
# img2 = cv2.resize(cv2.imread('0000011.jpg'), (256, 256)).astype(np.float32)[np.newaxis,...]
#
# real_img = tf.constant(img1)
# fake_img = tf.constant(img2)
#
# def compute_error(real, fake):
#     return tf.reduce_mean(tf.abs(fake - real))
# vgg_model_file = './imagenet-vgg-verydeep-19.mat'
#
# vgg_real = build_vgg19(real_img, vgg_model_file)
# vgg_fake = build_vgg19(fake_img, vgg_model_file)
#
# p0 = compute_error(vgg_real['input'], vgg_fake['input'])
# p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2']) / 2.6
# p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2']) / 4.8
# p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2']) / 3.7
# p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2']) / 5.6
# p5 = compute_error(vgg_real['conv5_2'], vgg_fake['conv5_2']) * 10 / 1.5
# total_loss = p0 + p1 + p2 + p3 + p4 + p5
#
# sess = tf.Session()
#
# loss = sess.run(total_loss)