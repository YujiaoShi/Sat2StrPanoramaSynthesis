import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import *
import numpy as np
import tensorflow_addons as tfa

def encoder_decoder(generator_inputs, generator_outputs_channels, ngf=4, activational_layer=tf.nn.softmax):
    layers = []

    # encoder_1: [batch, 512, 512, in_channels] => [batch, 256, 256, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, ngf)
        layers.append(output)

    layer_specs = [
        ngf * 2, # encoder_2: [batch, 256, 256, ngf] => [batch, 128, 128, ngf * 2]
        ngf * 4, # encoder_3: [batch, 128, 128, ngf * 2] => [batch, 64, 64, ngf * 4]
        ngf * 8, # encoder_4: [batch, 64, 64, ngf * 4] => [batch, 32, 32, ngf * 8]
        ngf * 8, # encoder_5: [batch, 32, 32, ngf * 8] => [batch, 16, 16, ngf * 8]
        ngf * 8, # encoder_6: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8, # encoder_7: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        # ngf * 8, # encoder_8: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        # (ngf * 8, 0.5),   # decoder_8: [batch, 1, 4, ngf * 8] => [batch, 2, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_7: [batch, 2, 8, ngf * 8 * 2] => [batch, 4, 16, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_6: [batch, 4, 16, ngf * 8 * 2] => [batch, 8, 32, ngf * 8 * 2]
        (ngf * 8, 0.0),   # decoder_5: [batch, 8, 32, ngf * 8 * 2] => [batch, 16, 64, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 16, 64, ngf * 8 * 2] => [batch, 32, 128, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 32, 128, ngf * 4 * 2] => [batch, 64, 256, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 64, 256, ngf * 2 * 2] => [batch, 128, 512, ngf * 2 * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            # if decoder_layer == 0:
            #     # first decoder layer doesn't have skip connections
            #     # since it is directly connected to the skip_layer
            #     input = layers[-1]
            # else:
            #     input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
            input = layers[-1]

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)
            layers.append(output)

    # decoder_1: [batch, 128, 512, ngf * 2] => [batch, 256, 1024, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        # input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(layers[-1])
        # output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.layers.conv2d_transpose(rectified, generator_outputs_channels, kernel_size=4, strides=(2, 2), padding="same",
                                            kernel_initializer=tf.zeros_initializer(),
                                            bias_initializer=tf.constant_initializer(
                                                np.concatenate([np.zeros(generator_outputs_channels - 1, dtype=np.float32),
                                                                np.ones(1, dtype=np.float32)], axis=0)))

        # output = tf.tanh(output)
        output = activational_layer(output*100)
        layers.append(output)

    return layers[-1]


def geometry_transform(aer_imgs, estimated_height, target_height, target_width, mode, grd_height, max_height,
                       method='column', geoout_type='image', dataset='CVUSA'):
    '''
    :param aer_imgs:
    :param estimated_height:
    :param mode: if estimated_height.channel ==1, type belongs to {'hole', 'column'};
           otherwise if estimated_height.channel>1, type belongs to {'radiusPlaneMethod', 'heightPlaneMethod'}
    The following two parameters are only needed if mode is 'radiusPlaneMethod'.
    :param method: select from {'column', 'point'}.
                    'column' means: for each point in overhead view, we poject it and the points under it to the grd view
                                    we use cusum to mimic this process
                    'point' means we only project the points in the overhead view image to the grd view.
    :param geoout_type: select from {'volume', 'image'}.
    :return:
    '''
    # PlaneNum = estimated_height.get_shape().as_list()[-1]
    # if height_channel==1:
    if mode=='heightPlaneMethod':
        output = MultiPlaneImagesAer2Grd_height(aer_imgs, estimated_height, target_height, target_width, grd_height,
                                       max_height, method, geoout_type, dataset)
    elif mode=='radiusPlaneMethod':
        output = MultiPlaneImagesAer2Grd_radius(aer_imgs, estimated_height, target_height, target_width,
                                                      grd_height, max_height, method, geoout_type, dataset)
    return output


def MultiPlaneImagesAer2Grd_height(signal, estimated_height, target_height, target_width, grd_height=-2, max_height=30,
                                   method='column', geoout_type='image', dataset='CVUSA'):
    PlaneNum = estimated_height.get_shape().as_list()[-1]

    if method == 'column':
        estimated_height = tf.cumsum(estimated_height, axis=-1)
        # the maximum plane corresponds to grd plane
    batch, S, _, channel = tf_shape(signal, 4)
    H, W, C = signal.get_shape().as_list()[1:]
    assert (H==W)

    i = np.arange(0, (target_height*2))
    j = np.arange(0, target_width)
    jj, ii = np.meshgrid(j, i)

    if dataset=='CVUSA':
        f = H/55
    elif dataset=='CVACT' or dataset=='CVACThalf':
        f = H/(50*206/256)
    elif dataset=='CVACTunaligned':
        f = H/50
    elif dataset=='OP':
        f = H/100

    # f = H/144

    tanii = np.tan(ii * np.pi / (target_height*2))

    images_list = []
    alphas_list = []

    # images_list_volume = []

    for i in range(PlaneNum):
        z = grd_height + (max_height-grd_height) * i/PlaneNum

        u_dup = -1 * np.ones([(target_height*2), target_width])
        v_dup = -1 * np.ones([(target_height*2), target_width])
        m = target_height

        v = S / 2. - f * z * tanii * np.sin(jj * 2 * np.pi / target_width)
        u = S / 2. + f * z * tanii * np.cos(jj * 2 * np.pi / target_width)

        if z < 0:
            u_dup[-m:, :] = u[-m:, :]
            v_dup[-m:, :] = v[-m:, :]
        else:
            u_dup[0:m, :] = u[0:m, :]
            v_dup[0:m:, :] = v[0:m:, :]

        n = int(target_height/2)

        uv = np.stack([v_dup[n:-n,...], u_dup[n:-n,...]], axis=-1)
        uv = uv.astype(np.float32)
        warp = tf.stack([uv]*batch, axis=0)

        # images_prob = tf.contrib.resampler.resampler(signal*estimated_height[..., i:i+1], warp)
        # images = tf.contrib.resampler.resampler(signal, warp)
        # alphas = tf.contrib.resampler.resampler(estimated_height[..., i:i + 1], warp)
        images = tfa.image.resampler(signal, warp)
        alphas = tfa.image.resampler(estimated_height[..., i:i + 1], warp)
        images_list.append(images)
        alphas_list.append(alphas)

        # images_list_volume.append(images_prob)

    if geoout_type == 'volume':

        return tf.concat([images_list[i]*alphas_list[i] for i in range(PlaneNum)], axis=-1)

        # return tf.concat(images_list, axis=-1) * tf.concat(alphas_list, axis=-1)  # shape = [batch, target_height, target_width, channel*PlaneNum]

    elif geoout_type == 'image':
        for i in range(PlaneNum):
            rgb = images_list[i]
            a = alphas_list[i]
            if i == 0:
                output = rgb * a
            else:
                rgb_by_alpha = rgb * a
                output = rgb_by_alpha + output * (1 - a)

        return output  # shape = [batch, target_height, target_width, channel]

    # batch_image = tf.stack(images_list, axis=-1)
    #
    # batch_mulplanes = tf.reshape(batch_image, [-1, target_height, target_width, C*PlaneNum])
    #
    # return batch_mulplanes


def MultiPlaneImagesAer2Grd_radius(signal, estimated_height, target_height, target_width, grd_height, max_height,
                                   method='column', geoout_type='image', dataset='CVUSA'):
    '''
    This function first convert uv coordinate to polar coordinate, i.e., from overhead planes to cylinder coordinate,
    and then from cylinder coordinate to spherical coordinate
    :param signal: [batch, height, width, channel] image
    :param estimated_height: [batch, height, width, PlaneNume]
    :param target_height: height/phi direction
    :param target_width: azimuth direction
    :param grd_height:
    :param max_height:
    :param method: select from {'column', 'point'}.
                    'column' means: for each point in overhead view, we poject it and the points under it to the grd view
                                    we use cusum to mimic this process
                    'point' means we only project the points in the overhead view image to the grd view.
    :param out_type: select from {'volume', 'image'}.
    :return:
    '''
    PlaneNum = estimated_height.get_shape().as_list()[-1]
    batch, height, width, channel = tf_shape(signal, rank=4)

    if method=='column':
        # estimated_height = tf.cumsum(estimated_height, axis=-1, reverse=True)
        # # the 0th plane corresponds to grd plane
        estimated_height = tf.cumsum(estimated_height, axis=-1)
        # the maximum plane corresponds to grd plane

    voxel = tf.transpose(tf.stack([signal]*PlaneNum, axis=-1), [0, 1, 2, 4, 3])
            # * tf.expand_dims(estimated_height, axis=-1)
    voxel = tf.reshape(voxel, [batch, height, width, PlaneNum*channel])

    ################### from overhead view uvz coordinate to cylinder pthetaz coordinate #########################
    S = signal.get_shape().as_list()[1]
    radius = int(S//4)
    azimuth = target_width

    i = np.arange(0, radius)
    j = np.arange(0, azimuth)
    jj, ii = np.meshgrid(j, i)

    # if train_mode:
    #     sx = np.random.uniform(-10, 10)
    #     sy = np.random.uniform(-10, 10)
    #     rx = np.minimum(S/2.-sx, S/2.+sx)
    #     ry = np.minimum(S/2.-sy, S/2.+sy)
    #
    #     y = (S / 2. + sx) - rx / radius * (radius - 1 - ii) * np.sin(2 * np.pi * jj / azimuth)
    #     x = (S / 2. + sy) + ry / radius * (radius - 1 - ii) * np.cos(2 * np.pi * jj / azimuth)
    #
    # else:

    y = S / 2. - S / 2. / radius * (radius - 1 - ii) * np.sin(2 * np.pi * jj / azimuth)
    x = S / 2. + S / 2. / radius * (radius - 1 - ii) * np.cos(2 * np.pi * jj / azimuth)

    uv = np.stack([y, x], axis=-1)
    uv = uv.astype(np.float32)
    warp = tf.stack([uv] * batch, axis=0)

    # imgs = tf.contrib.resampler.resampler(voxel, warp)
    imgs = tfa.image.resampler(voxel, warp)
    imgs = tf.reshape(imgs, [batch, radius, azimuth, PlaneNum, channel])  # batch, radius, azimuth, PlaneNum, channel]
    # imgs = tf.transpose(imgs, [0, 3, 2, 1, 4])[:, ::-1, ...]
    # # shape = [batch, PlaneNum, azimuth, radius, channel]
    # # the maximum PlaneNum corresponds to ground plane
    # alpha = tf.contrib.resampler.resampler(estimated_height, warp)[..., ::-1]  # batch, radius, azimuth, PlaneNum
    # # the maximum PlaneNum corresponds to ground plane
    # alpha = tf.transpose(alpha, [0, 3, 2, 1])  # shape = [batch, PlaneNum, azimuth, radius]
    imgs = tf.transpose(imgs, [0, 3, 2, 1, 4])
    # shape = [batch, PlaneNum, azimuth, radius, channel]
    # the maximum PlaneNum corresponds to ground plane
    # alpha = tf.contrib.resampler.resampler(estimated_height, warp)  # batch, radius, azimuth, PlaneNum
    alpha = tfa.image.resampler(estimated_height, warp)
    # the maximum PlaneNum corresponds to ground plane
    alpha = tf.transpose(alpha, [0, 3, 2, 1])  # shape = [batch, PlaneNum, azimuth, radius]

    if dataset == 'CVUSA':
        meters = 55
    elif dataset == 'CVACT' or dataset=='CVACThalf':
        meters = (50 * 206 / 256)
    elif dataset == 'CVACTunaligned':
        meters = 50
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

    if geoout_type=='volume':

        return tf.concat([rgb_layers[i]*a_layers[i] for i in range(radius)], axis=-1)

        # return tf.concat(rgb_layers[::-1], axis=-1) * tf.concat(a_layers[::-1], axis=-1) # shape = [batch, target_height, target_width, channel*PlaneNum]

    elif geoout_type=='image':
        for i in range(radius):
            rgb = rgb_layers[i]
            a = a_layers[i]
            if i==0:
                output = rgb * a
            else:
                rgb_by_alpha = rgb * a
                output = rgb_by_alpha + output * (1 - a)

        return output  # shape = [batch, target_height, target_width, channel]




















# def geometry_transform_hole(aer_imgs, estimated_height, target_height, target_width, grd_height=-2.5, max_height=47.5):
#     _, aer_size, _, heightPlaneNum = estimated_height.get_shape().as_list()
#     batch, _, _, channel = tf_shape(aer_imgs, 4)
#
#     f = 144/aer_size
#
#     assert heightPlaneNum==1
#     estimated_height = tf.squeeze(estimated_height) # shape = [batch, aer_size, aer_size]
#
#     estimated_height = grd_height + (max_height - grd_height) * estimated_height
#
#     i = np.arange(0, aer_size)
#     j = np.arange(0, aer_size)
#     jj, ii = np.meshgrid(j, i)
#
#     radius = np.sqrt((ii - (aer_size / 2 - 0.5)) ** 2 + (jj - (aer_size / 2 - 0.5)) ** 2)
#
#     Theta1 = tf.atan(
#         (ii[:, 0:int(aer_size / 2)] - (aer_size / 2 - 0.5)) / (jj[:, 0:int(aer_size / 2)] - (aer_size / 2 - 0.5))) + 0.5 * np.pi
#     Theta2 = tf.atan(
#         (ii[:, int(aer_size / 2):] - (aer_size / 2 - 0.5)) / (jj[:, int(aer_size / 2):] - (aer_size / 2 - 0.5))) + 1.5 * np.pi
#     Theta = tf.concat([Theta1, Theta2], axis=-1)
#
#     Phimax = tf.atan2(radius, estimated_height*f)
#     Phimin = tf.atan2(radius, grd_height*f)
#
#     Theta = Theta / 2 / np.pi * target_width         # shape = [aer_size, aer_size]
#     Phimax = Phimax / np.pi * (target_height * 2)    # shape = [aer_size, aer_size]
#     Phimin = Phimin / np.pi * (target_height * 2)    # shape = [aer_size, aer_size]
#
#     target = tf.zeros([batch, target_height*2, target_width, channel])
#
#     for rr in range(aer_size//2):
#
#         r = aer_size//2 - rr
#
#         indices = tf.where((radius > (r-1)) & (radius <= r)) # shape = [batch*num, 3] 3--> batch, height, width
#
#         selected_Theta = tf.gather_nd(Theta, indices)       # shape = [batch*num]
#         selected_Phimax = tf.gather_nd(Phimax, indices)     # shape = [batch*num]
#         selected_Phimin = tf.gather_nd(Phimin, indices)     # shape = [batch*num]
#
#         rgb = tf.gather_nd(aer_imgs, indices)               # shape = [batch*num, channel] channel = aer_imgs.shape[-1]
#
#         iy = tf.minimum(tf.cast(tf.round(selected_Theta), tf.int64), target_width-1)           # shape = [batch*num]
#         ix1 = tf.minimum(tf.cast(tf.round(selected_Phimax), tf.int64), target_height*2-1)      # shape = [batch*num]
#         ix0 = tf.minimum(tf.cast(tf.round(selected_Phimin), tf.int64), target_height*2 - 1)    # shape = [batch*num]
#
#         num = tf_shape(iy, 1)//batch
#         bi = tf.reshape(tf.range(batch), [1, batch])
#         bi = tf.tile(bi, [num, 1])
#         bi = tf.reshape(bi, [-1])   # shape = [batch*num]
#         index = tf.stack([bi, ix1, iy], axis=-1)  # shape = [batch*num, 3] 3-->batch, phi, theta
#
#         assign = tf.assign(tf.gather_nd(target, index), rgb)













