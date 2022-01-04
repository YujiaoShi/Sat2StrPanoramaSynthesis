
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import PIL.Image as Image
import tensorflow_addons as tfa

dataset='OP'
# img_root = '../../../Data/CVUSA/bingmap/19/'
img_root = '../../../Data/OP/aerial/'
# img_root = '../../../Data/CVACT/satview_correct/'
files = os.listdir(img_root)
file = os.path.join(img_root, files[0])
print(file)
# img = np.asarray(Image.open(file).resize((256, 256)), np.float32)[None, ...] # [1, 256, 256, 3]
img = np.asarray(Image.open(file), np.float32)[None, ...] # [1, 256, 256, 3]
signal = tf.constant(img)
# estimated_height = tf.ones_like(signal[..., 0:1])
# estimated_height = tf.ones(signal.get_shape().as_list()[:-1] + [1], dtype=tf.float32)
estimated_height = tf.concat([tf.zeros(signal.get_shape().as_list()[:-1] + [63]),
                              tf.ones(signal.get_shape().as_list()[:-1] + [1])], axis=-1)
target_height = 512 #128
target_width = 512 * 2
grd_height = -2 
max_height = 32
method='column'
geoout_type='image'

PlaneNum = estimated_height.get_shape().as_list()[-1]
batch, height, width, channel = signal.get_shape().as_list()

if method=='column':
    # estimated_height = tf.cumsum(estimated_height, axis=-1, reverse=True)
    # # the 0th plane corresponds to grd plane
    estimated_height = tf.cumsum(estimated_height, axis=-1)
    # the maximum plane corresponds to grd plane

voxel = tf.transpose(tf.stack([signal]*PlaneNum, axis=-1), [0, 1, 2, 4, 3])
       
voxel = tf.reshape(voxel, [batch, height, width, PlaneNum*channel])

################### from overhead view uvz coordinate to cylinder pthetaz coordinate #########################
S = signal.get_shape().as_list()[1]
radius = int(S//4)
azimuth = target_width

i = np.arange(0, radius)
j = np.arange(0, azimuth)
jj, ii = np.meshgrid(j, i)


y = S / 2. - S / 2. / radius * (radius - 1 - ii) * np.sin(2 * np.pi * jj / azimuth)
x = S / 2. + S / 2. / radius * (radius - 1 - ii) * np.cos(2 * np.pi * jj / azimuth)

uv = np.stack([y, x], axis=-1)
uv = uv.astype(np.float32)
warp = tf.stack([uv] * batch, axis=0)

# imgs = tf.contrib.resampler.resampler(voxel, warp)
imgs = tfa.image.resampler(voxel, warp)
imgs = tf.reshape(imgs, [batch, radius, azimuth, PlaneNum, channel])  # batch, radius, azimuth, PlaneNum, channel]
imgs = tf.transpose(imgs, [0, 3, 2, 1, 4])
# shape = [batch, PlaneNum, azimuth, radius, channel]
# the maximum PlaneNum corresponds to ground plane

alpha = tfa.image.resampler(estimated_height, warp)
# the maximum PlaneNum corresponds to ground plane
alpha = tf.transpose(alpha, [0, 3, 2, 1])  # shape = [batch, PlaneNum, azimuth, radius]

if dataset == 'CVUSA':
    meters = 55
elif dataset == 'CVACT' or dataset=='CVACThalf':
    meters = (50 * 206 / 256)
elif dataset == 'OP':
    meters = 100

################### from cylinder pthetaz coordinate to grd phithetar coordinate #########################
    if dataset=='CVUSA' or dataset=='CVACThalf':
        i = np.arange(0, target_height*2)
        j = np.arange(0, target_width)
        jj, ii = np.meshgrid(j, i)
        tanPhi = np.tan(ii / target_height / 2 * np.pi)
        tanPhi[np.where(tanPhi==0)] = 1e-16

        n = int(target_height//2)

        MetersPerRadius = meters / 2 / radius
        rgb_layers = []
        a_layers = []
        for r in range(0, radius):
            # from far to near
            z = (radius-r-1)*MetersPerRadius/tanPhi[n:-n]
            z = (PlaneNum-1) - (z - grd_height)/(max_height - grd_height) * (PlaneNum-1)
            theta = jj[n:-n]
            uv = np.stack([theta, z], axis=-1)
            uv = uv.astype(np.float32)
            warp = tf.stack([uv] * batch, axis=0)
            # rgb = tf.contrib.resampler.resampler(imgs[..., r, :], warp)
            rgb = tfa.image.resampler(imgs[..., r, :], warp)
            # a = tf.contrib.resampler.resampler(alpha[..., r:r + 1], warp)
            a = tfa.image.resampler(alpha[..., r:r+1], warp)

            rgb_layers.append(rgb)
            a_layers.append(a)

    else:
        i = np.arange(0, target_height)
        j = np.arange(0, target_width)
        jj, ii = np.meshgrid(j, i)
        tanPhi = np.tan(ii / target_height * np.pi)
        tanPhi[np.where(tanPhi == 0)] = 1e-16

        # n = int(target_height // 2)

        MetersPerRadius = meters / 2 / radius
        rgb_layers = []
        a_layers = []
        for r in range(0, radius):
            # from far to near
            z = (radius - r - 1) * MetersPerRadius / tanPhi
            z = (PlaneNum - 1) - (z - grd_height) / (max_height - grd_height) * (PlaneNum - 1)
            theta = jj
            uv = np.stack([theta, z], axis=-1)
            uv = uv.astype(np.float32)
            warp = tf.stack([uv] * batch, axis=0)
            # rgb = tf.contrib.resampler.resampler(imgs[..., r, :], warp)
            # a = tf.contrib.resampler.resampler(alpha[..., r:r + 1], warp)
            rgb = tfa.image.resampler(imgs[..., r, :], warp)
            a = tfa.image.resampler(alpha[..., r:r + 1], warp)

            rgb_layers.append(rgb)
            a_layers.append(a)


for i in range(radius):
    rgb = rgb_layers[i]
    a = a_layers[i]
    if i==0:
        output = rgb * a
    else:
        rgb_by_alpha = rgb * a
        output = rgb_by_alpha + output * (1 - a)

sess = tf.Session()

img0, alpha0 = sess.run([imgs, alpha])
img1_list = sess.run(rgb_layers)
alpha1_list = sess.run(a_layers)

iimg = Image.fromarray(img0[0,0].astype(np.uint8))
iimg.save('stage1.png')
out_img = sess.run(output)
iimg = Image.fromarray(out_img[0].astype(np.uint8))
iimg.save('stage2.png')

a = 1
# for idx in range(img0.shape[0]):

   