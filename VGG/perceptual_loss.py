
import tensorflow as tf
from .vgg import build_vgg19



def compute_error(real, fake):
    return tf.reduce_mean(tf.abs(fake - real))


def perceptual_loss(real_img, fake_img):
    real_img = (real_img+1.)/2. * 255.
    fake_img = (fake_img+1.)/2. * 255.
    vgg_model_file = '../VGG/imagenet-vgg-verydeep-19.mat'

    vgg_real = build_vgg19(real_img, vgg_model_file)
    vgg_fake = build_vgg19(fake_img, vgg_model_file)

    p0 = compute_error(vgg_real['input'], vgg_fake['input'])
    p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2']) / 2.6
    p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2']) / 4.8
    p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2']) / 3.7
    p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2']) / 5.6
    p5 = compute_error(vgg_real['conv5_2'], vgg_fake['conv5_2']) * 10 / 1.5
    total_loss = p0 + p1 + p2 + p3 + p4 + p5

    return total_loss


def perceptual_loss_n(real_img, fake_imgs):

    vgg_model_file = '../VGG/imagenet-vgg-verydeep-19.mat'

    real_img = (real_img + 1.) / 2. * 255.
    vgg_real = build_vgg19(real_img, vgg_model_file)

    loss = []

    for fake in fake_imgs:
        fake = (fake + 1.) / 2. * 255.

        vgg_fake = build_vgg19(fake, vgg_model_file)

        p0 = compute_error(vgg_real['input'], vgg_fake['input'])
        p1 = compute_error(vgg_real['conv1_2'], vgg_fake['conv1_2']) / 2.6
        p2 = compute_error(vgg_real['conv2_2'], vgg_fake['conv2_2']) / 4.8
        p3 = compute_error(vgg_real['conv3_2'], vgg_fake['conv3_2']) / 3.7
        p4 = compute_error(vgg_real['conv4_2'], vgg_fake['conv4_2']) / 5.6
        p5 = compute_error(vgg_real['conv5_2'], vgg_fake['conv5_2']) * 10 / 1.5

        loss.append(p0 + p1 + p2 + p3 + p4 + p5)

    total_loss = tf.stack(loss)
    min_loss = tf.reduce_min(total_loss)

    return min_loss

