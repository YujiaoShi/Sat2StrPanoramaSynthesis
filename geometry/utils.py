import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from tensorflow.python.ops import math_ops


def softargmax(x, beta=100):
    x_range = tf.range(x.shape.as_list()[-1], dtype=tf.float32)
    return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1, keep_dims=True)


def tf_shape(x, rank):
    static_shape = x.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(x), rank)
    return [s if s is not None else d for s,d in zip(static_shape, dynamic_shape)]


def safe_divide(numerator, denominator, name='safe_divide'):
    return tf.where(math_ops.greater(denominator, 0), math_ops.divide(numerator, denominator), tf.zeros_like(numerator)
                    , name=name)


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def deprocess_label(label_logits):
    '''
    :param label_logits: label.shape = [batch, height, width, 4] --> 4 is label number, value from 0 to 1
    :return: label: shape =[batch, height, width, 3] value in {0, 255}, for the purpose of show.
    '''
    label_onehot = tf.one_hot(tf.argmax(label_logits, axis=-1), depth=4)
    label = label_onehot[..., 1:]*255
    return label



def warp_pad_columns(x, n=1):

    out = tf.concat([x[:, :, -n:, :], x, x[:, :, :n, :]], axis=2)
    return tf.pad(out, [[0, 0], [n, n], [0, 0], [0, 0]])


def conv_layer_cir(x, kernel_dim, strides, output_dim, trainable, activated, bn,
                   name='layer_conv', activation_function=tf.nn.relu):
    n = int((kernel_dim - 1) / 2)
    x = warp_pad_columns(x, n)

    input_dim = x.get_shape().as_list()[-1]
    with tf.variable_scope(name): # reuse=tf.AUTO_REUSE
        weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                 trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name='biases', shape=[output_dim],
                               trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

        out = tf.nn.conv2d(x, weight, strides, padding='VALID') + bias

        if bn:
            out = batchnorm(out)

        if activated:
            out = activation_function(out)

        return out



def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))



def gen_conv(batch_input, out_channels, separable_conv=False):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels, separable_conv=False):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a=0.2):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def corr_distance_orien_unknow(grd_matrix, sat_matrix):
    '''
    correlation distance for localizing ground panoramas with unknown orientation
    :param grd_matrix: shape = [batch_grd, height, grd_width, channel]
    :param sat_matrix: shape = [batch_sat, height, sat_width, channel]
    :return:
    '''
    try:
        grd_batch, grd_height, grd_width, grd_channel = grd_matrix.get_shape().as_list()
        sat_batch, sat_height, sat_width, sat_channel = sat_matrix.get_shape().as_list()
    except:
        grd_batch, grd_height, grd_width, grd_channel = grd_matrix.shape
        sat_batch, sat_height, sat_width, sat_channel = sat_matrix.shape

    assert grd_height==sat_height, grd_channel==sat_channel

    def warp_pad_columns(x, n):
        out = tf.concat([x, x[:, :, :n, :]], axis=2)
        return out

    n = grd_width - 1
    x = warp_pad_columns(sat_matrix, n)

    weight = tf.transpose(grd_matrix, [1, 2, 3, 0])

    out = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')

    assert out.get_shape().as_list() == [sat_batch, 1, sat_width, grd_batch]

    out = tf.squeeze(out)  # shape = [sat_batch, sat_width, grd_batch]

    ############################ ground truth orientation corresponded distance ###############################


    max_dis = 2 - 2 * tf.transpose(tf.reduce_max(out, axis=1))  # shape = [grd_batch, sat_batch]

    pred_orien = tf.diag_part(tf.argmax(out, axis=1))  # shape = [sat_batch, grd_batch]

    return max_dis, pred_orien


def triplet_loss(grd_matrix, sat_matrix, batch_size):
    '''
    :param grd_matrix: shape = [grd_batch, grd_height, grd_width, grd_channel]
    :param sat_matrix: shape = [sat_batch, sat_height, sat_width, sat_channel]
                       grd_batch==sat_batch grd_height==sat_height grd_channel==sat_channel grd_width<=sat_width
    :param grd_orien:  shape = [grd_batch] the north direction (value within 0~sat_width) of each grd image
    :param train_grd_noise:
    :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
    :param train_method: 0: triplet(max_dis) + regularize * (max_dis - orien_dis)
                         1: triplet(orien_dis) + regularize * (max_dis - orien_dis)
    :param regularize:
    :return:
    '''

    with tf.name_scope('weighted_soft_margin_triplet_loss'):

        dist_array, pred_orien = corr_distance_orien_unknow(grd_matrix, sat_matrix)

        pos_dist = tf.diag_part(dist_array)

        pair_n = batch_size * (batch_size - 1.0)

        # ground to satellite
        triplet_dist_g2s = pos_dist - dist_array
        loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * 10))) / pair_n

        # satellite to ground
        triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
        loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * 10))) / pair_n

        loss = (loss_g2s + loss_s2g) / 2.0

    return loss


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
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 512, ngf * 2] => [batch, 256, 1024, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        # input = tf.concat([layers[-1], layers[0]], axis=3) tf.random_normal_initializer(0, 0.02)
        rectified = tf.nn.relu(layers[-1])
        # output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.layers.conv2d_transpose(rectified, generator_outputs_channels, kernel_size=4, strides=(2, 2),
                                            padding="same",
                                            kernel_initializer=tf.zeros_initializer(),
                                            bias_initializer=tf.constant_initializer(
                                                np.concatenate(
                                                    [np.zeros(generator_outputs_channels - 1, dtype=np.float32),
                                                     np.ones(1, dtype=np.float32)], axis=0)))
        # output = tf.tanh(output)
        output = activational_layer(output)
        layers.append(output)

    return layers[-1]





# def sample_within_bounds_xyz(signal, batch_index, x, y, z, channel_index):
#     '''
#     :param signal: tf variable, shape = [batch, height, width, PlaneNum, channel]
#     :param x: numpy
#     :param y: numpy
#     :return:
#     '''
#
#     index = tf.stack([tf.reshape(batch_index, [-1]), tf.reshape(x, [-1]), tf.reshape(y, [-1]),
#                       tf.reshape(z, [-1]), tf.reshape(channel_index, [-1])], axis=1)
#
#     result = tf.gather_nd(signal, index)
#
#     batch, height, width, channel = tf_shape(x, rank=4)
#
#     sample = tf.reshape(result, [batch, height, width, channel])
#
#     return sample
#
#
# def sample_bilinear_xyz(signal, batch_index, rx, ry, rz, channel_index):
#     '''
#     :param signal: tensor_shape = [batch, sat_height, sat_width, heightPlaneNum, channel]
#     :param rx: tensor_shape = [batch, grd_height, grd_width, channel]
#     :param ry: tensor_shape = [batch, grd_height, grd_width, channel]
#     :param batch_index: tensor_shape = [batch, grd_height, grd_width, channel]
#     :param channel_index: tensor_shape = [batch, grd_height, grd_width, channel]
#     :return:
#     '''
#
#     signal_dim_x, signal_dim_y, signal_dim_z = signal.get_shape().as_list()[1:-1]
#
#     # obtain four sample coordinates
#     ix0 = tf.maximum(tf.cast(rx, tf.int32), 0)
#     iy0 = tf.maximum(tf.cast(ry, tf.int32), 0)
#     iz0 = tf.maximum(tf.cast(rz, tf.int32), 0)
#
#     ix1 = tf.minimum(ix0 + 1, signal_dim_x-1)
#     iy1 = tf.minimum(iy0 + 1, signal_dim_y-1)
#     iz1 = tf.minimum(iz0 + 1, signal_dim_z-1)
#
#     # sample signal at each four positions
#     signal_000 = sample_within_bounds_xyz(signal, batch_index, ix0, iy0, iz0, channel_index)
#     signal_100 = sample_within_bounds_xyz(signal, batch_index, ix0, iy1, iz0, channel_index)
#     signal_010 = sample_within_bounds_xyz(signal, batch_index, ix1, iy0, iz0, channel_index)
#     signal_110 = sample_within_bounds_xyz(signal, batch_index, ix1, iy1, iz0, channel_index)
#
#     signal_001 = sample_within_bounds_xyz(signal, batch_index, ix0, iy0, iz1, channel_index)
#     signal_101 = sample_within_bounds_xyz(signal, batch_index, ix0, iy1, iz1, channel_index)
#     signal_011 = sample_within_bounds_xyz(signal, batch_index, ix1, iy0, iz1, channel_index)
#     signal_111 = sample_within_bounds_xyz(signal, batch_index, ix1, iy1, iz1, channel_index)
#
#     ix1 = tf.cast(ix1, tf.float32)
#     iy1 = tf.cast(iy1, tf.float32)
#     iz1 = tf.cast(iz1, tf.float32)
#
#     fx00 = (ix1 - rx) * signal_100 + (rx - ix0) * signal_000
#     fx10 = (ix1 - rx) * signal_110 + (rx - ix0) * signal_010
#     fy0 = (iy1 - ry) * fx10 + (ry - iy0) * fx00
#
#     fx01 = (ix1 - rx) * signal_101 + (rx - ix0) * signal_001
#     fx11 = (ix1 - rx) * signal_111 + (rx - ix0) * signal_011
#     fy1 = (iy1 - ry) * fx11 + (ry - iy0) * fx01
#
#     fz = (iz1 - rz) * fy1 + (rz - iz0) * fy0
#
#     return fz
#
#
#
# def MultiPlaneImagesAer2Grd_radius(signal, estimated_height, target_height, target_width, grd_height, max_height):
#     '''
#     :param x: tf variable, x.shape=[batch, S, S, channel]
#     :param height: output height
#     :param width: output width
#     :param radius: shape = [batch, height, width, channel] its value is within the range of [0, S/2).
#     :return:
#     '''
#     batch, S, _, channel = tf_shape(signal, 4)
#     PlaneNum = estimated_height.get_shape().as_list()[-1]  # shape = [batch, S, S, PlaneNum]
#
#     Voxel = tf.transpose(tf.stack([signal]*PlaneNum, axis=-1), [0, 1, 2, 4, 3])  # shape = [batch, S, S, PlaneNum, channel]
#     Voxel = tf.expand_dims(estimated_height, axis=-1) * Voxel  # shape = [batch, S, S, PlaneNum, channel]
#
#     f = 144/S
#
#     b = tf.range(0, batch)
#     h = tf.range(0, target_height*2)
#     w = tf.range(0, target_width)
#     c = tf.range(0, channel)
#
#     bb, hh, ww, cc = tf.meshgrid(b, h, w, c, indexing='ij')
#
#     sinTheta = tf.sin(ww / target_width * np.pi * 2)
#     cosTheta = tf.cos(ww / target_width * np.pi * 2)
#     tanPhi = tf.tan(hh / (target_height * 2) * np.pi)
#
#     ww = tf.cast(ww, tf.float32)
#     RadiusNum = int(signal.get_shape().as_list()[1] / 2)
#
#     target_volume = []
#     for r in range(1, RadiusNum):
#         # r = RadiusNum - i
#         x = S/2 + r * cosTheta
#         y = S/2 + r * sinTheta
#         z = safe_divide(r * f, tanPhi)
#         z = (z - grd_height)/(max_height - grd_height) * PlaneNum
#
#         sample = sample_bilinear_xyz(Voxel, bb, x, y, z, cc)
#         target_volume.append(sample)








# def warp_pad_columns(x, m1=1, m2=1, n1=1, n2=1):
#     out = tf.concat([x[:, :, -n1:, :], x, x[:, :, :n2, :]], axis=2)
#     return tf.pad(out, [[0, 0], [m1, m2], [0, 0], [0, 0]])
#
#
# def discrim_conv_cir(batch_input, out_channels, stride):
#     padded_input = warp_pad_columns(batch_input, m1=1, m2=1, n1=1, n2=1)
#     return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
#                             kernel_initializer=tf.random_normal_initializer(0, 0.02))
#
#
# def gen_conv_cir(batch_input, out_channels):
#     initializer = tf.random_normal_initializer(0, 0.02)
#     x = warp_pad_columns(batch_input, m1=1, m2=1, n1=1, n2=1)
#     return tf.layers.conv2d(x, out_channels, kernel_size=4, strides=(2, 2), padding="valid", kernel_initializer=initializer)
#
#
# def gen_deconv_cir(batch_input, out_channels):
#     initializer = tf.random_normal_initializer(0, 0.02)
#     _, height, width, channel = batch_input.get_shape().as_list()
#     x = tf.image.resize_nearest_neighbor(batch_input, (2*height, 2*width))
#     x = warp_pad_columns(x, m1=1, m2=1, n1=1, n2=1)
#     return tf.layers.conv2d(x, out_channels, kernel_size=3, strides=(1,1), padding="valid", kernel_initializer=initializer)
