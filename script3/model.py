import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import collections
from VGG.perceptual_loss import perceptual_loss
from geometry.Geometry import *
from geometry.projector import *
from geometry.utils import *


EPS = 1e-7

target_height = 128
target_width = 512
aer_size = 256
grd_height = -2
max_height = 6

Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, "
                                        "estimated_height, generator_inputs,"
                                        "gen_loss_GAN, gen_loss_L1, gen_loss_perceptual, gen_grads_and_vars, train")

def create_generator(generator_inputs, ref_images, a):

    if a.finalout_type == 'image':  # ['image', 'rgba', 'fgbg']
        generator_outputs_channels = 3
    elif a.finalout_type == 'rgba':
        generator_outputs_channels = a.radiusPlaneNum * 4
    elif a.finalout_type == 'fgbg':
        generator_outputs_channels = a.radiusPlaneNum * 2 + 3

    ngf = a.ngf
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

    bottleneck = tf.reshape(output, [-1, 1, 4, output.get_shape().as_list()[-1]])
    layers.append(bottleneck)

    layer_specs = [
        # (ngf * 8, 0.5),   # decoder_8: [batch, 1, 4, ngf * 8] => [batch, 2, 8, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_7: [batch, 2, 8, ngf * 8 * 2] => [batch, 4, 16, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_6: [batch, 4, 16, ngf * 8 * 2] => [batch, 8, 32, ngf * 8 * 2]
        (ngf * 8, 0.5),   # decoder_5: [batch, 8, 32, ngf * 8 * 2] => [batch, 16, 64, ngf * 8 * 2]
        (ngf * 4, 0.0),   # decoder_4: [batch, 16, 64, ngf * 8 * 2] => [batch, 32, 128, ngf * 4 * 2]
        (ngf * 2, 0.0),   # decoder_3: [batch, 32, 128, ngf * 4 * 2] => [batch, 64, 256, ngf * 2 * 2]
        (ngf, 0.0),       # decoder_2: [batch, 64, 256, ngf * 2 * 2] => [batch, 128, 512, ngf * 2 * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):

            if a.skip:

                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer-1]], axis=3)
            else:

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
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    if a.finalout_type == 'image':  # ['image', 'rgba', 'fgbg']
        outputs_grd = layers[-1]

        return outputs_grd

    elif a.finalout_type == 'rgba':
        outputs_grd = mpi_render_grd_view(layers[-1], share_alpha=True)
        outputs_grd = tf.tanh(outputs_grd)
        render_aer = mpi_render_aer_view(layers[-1], share_alpha=True)
        # render_aer = rtheta2uv(render_aer, a.radiusPlaneNum * 2)
        # outputs_aer = refine(render_aer)
        render_aer = rtheta2uv(render_aer, 256)
        with tf.variable_scope('refine_aer'):
            outputs_aer = encoder_decoder(render_aer, 3, ngf=16, activational_layer=tf.nn.tanh)

        return outputs_grd, outputs_aer

    elif a.finalout_type == 'fgbg':
        blend_weights = (layers[-1][:, :, :, :a.radiusPlaneNum] + 1.) / 2.
        alphas = (layers[-1][:, :, :, a.radiusPlaneNum: 2 * a.radiusPlaneNum] + 1.) / 2.
        bg_rgb = layers[-1][..., -3:]
        fg_rgb = ref_images

        for i in range(a.radiusPlaneNum):
            curr_alpha = tf.expand_dims(alphas[:, :, :, i], -1)
            w = tf.expand_dims(blend_weights[:, :, :, i], -1)
            curr_rgb = w * fg_rgb + (1 - w) * bg_rgb
            curr_rgba = tf.concat([curr_rgb, curr_alpha], axis=3)
            if i == 0:
                rgba_layers = curr_rgba
            else:
                rgba_layers = tf.concat([rgba_layers, curr_rgba], axis=3)

        outputs_grd = mpi_render_grd_view(rgba_layers, share_alpha=True)
        render_aer = mpi_render_aer_view(rgba_layers, share_alpha=True)
        # render_aer = rtheta2uv(render_aer, a.radiusPlaneNum * 2)
        # outputs_aer = refine(render_aer)
        render_aer = rtheta2uv(render_aer, 256)
        with tf.variable_scope('refine_aer'):
            outputs_aer = encoder_decoder(render_aer, 3, ngf=4, activational_layer=tf.nn.tanh)

        return outputs_grd, outputs_aer


def create_discriminator(discrim_inputs, ndf=64):
    n_layers = 3
    layers = []

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = discrim_conv(discrim_inputs, ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discrim_conv(layers[-1], out_channels, stride=stride)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = discrim_conv(rectified, out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets, ref_images, a):

    with tf.variable_scope("generator"):
        
        with tf.variable_scope('height_estimation'):

            if a.heightPlaneNum > 1:

                estimated_height = encoder_decoder(inputs, generator_outputs_channels=a.heightPlaneNum, ngf=4,
                                                     activational_layer=tf.nn.softmax)
            else:
                estimated_height = tf.concat([tf.zeros(inputs.get_shape().as_list()[:-1] + [63]),
                                              tf.ones(inputs.get_shape().as_list()[:-1] + [1])], axis=-1)

        generator_inputs = geometry_transform(inputs, estimated_height, target_height, target_width,
                                              a.height_mode, grd_height, max_height, a.method, a.geoout_type, a.dataset)

        # height, width = targets.get_shape().as_list()[1:-1]
        # concat_inputs = tf.concat([generator_inputs[:, : int(height/2), :, :], ref_images[:, int(height/2):, :, :]], axis=1)

        outputs = create_generator(generator_inputs, ref_images, a)

        if a.finalout_type == 'image':  # ['image', 'rgba', 'fgbg']
            outputs_grd = outputs

        else:
            outputs_grd, outputs_aer = outputs

    with tf.name_scope("real_discriminator_grd"):
        with tf.variable_scope("discriminator_grd"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real_grd = create_discriminator(targets)

    with tf.name_scope("fake_discriminator_grd"):
        with tf.variable_scope("discriminator_grd", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake_grd = create_discriminator(outputs_grd)


    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = 0.5 * (tf.reduce_mean(-(tf.log(predict_real_grd + EPS) + tf.log(1 - predict_fake_grd + EPS))))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN_grd = tf.reduce_mean(-tf.log(predict_fake_grd + EPS))
        gen_loss_L1_grd = tf.reduce_mean(tf.abs(targets - outputs_grd))
        gen_loss_perceptual_grd = perceptual_loss(targets, outputs_grd)
        if a.finalout_type != 'image':
            gen_loss_L1_aer = tf.reduce_mean(tf.abs(inputs - outputs_aer))
            gen_loss_perceptual_aer = perceptual_loss(inputs, outputs_aer)
            gen_loss = gen_loss_GAN_grd * a.gan_weight + \
                       gen_loss_perceptual_grd * a.perceptual_weight_grd + \
                       gen_loss_perceptual_aer * a.perceptual_weight_aer + \
                       gen_loss_L1_grd * a.l1_weight_grd + \
                       gen_loss_L1_aer * a.l1_weight_aer
        else:

            gen_loss = gen_loss_GAN_grd * a.gan_weight + \
                       gen_loss_perceptual_grd * a.perceptual_weight_grd + \
                       gen_loss_L1_grd * a.l1_weight_grd

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN_grd, gen_loss_L1_grd, gen_loss_perceptual_grd])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real_grd,
        predict_fake=predict_fake_grd,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN_grd),
        gen_loss_L1=ema.average(gen_loss_L1_grd),
        gen_loss_perceptual=ema.average(gen_loss_perceptual_grd),
        gen_grads_and_vars=gen_grads_and_vars,
        # estimated_height=tf.argmax(estimated_height, axis=-1),
        estimated_height=estimated_height,
        generator_inputs=generator_inputs,
        outputs=outputs_grd,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )

