from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np
import argparse
import os
import json

import random
import collections
import math
import time
import PIL.Image as Image

from model23 import *

parser = argparse.ArgumentParser()
# parser.add_argument("--input_dir", help="path to folder containing images", default='facades/train')
parser.add_argument("--dataset", help="dataset", default='CVUSA')
parser.add_argument("--mode", choices=["train", "test", "export"], default="train")
parser.add_argument("--output_dir", help="where to put output files", default='pix2pix_perceploss')
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default=35, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=4, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoG", choices=["AtoG", "GtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")

parser.add_argument("--inputs_type", choices=["original", "geometry"], default="geometry")

parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--l1_weight_grd", type=float, default=0.0, help="weight on GAN term for generator gradient")
parser.add_argument("--l1_weight_aer", type=float, default=10.0, help="weight on L1 term for generator gradient")
parser.add_argument("--perceptual_weight_grd", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--perceptual_weight_aer", type=float, default=0.0, help="weight on GAN term for generator gradient")

parser.add_argument("--heightPlaneNum", type=int, default=32, help="weight on GAN term for generator gradient")
parser.add_argument("--radiusPlaneNum", type=int, default=32, help="weight on GAN term for generator gradient")
parser.add_argument("--height_mode", choices=['radiusPlaneMethod', 'heightPlaneMethod'], default='radiusPlaneMethod')
# Only when 'height_mode' is 'radiusPlaneMethod', the following two parameters are required. Otherwise not.
parser.add_argument("--method", choices=['column', 'point'], default='column')
parser.add_argument("--geoout_type", choices=['volume', 'image'], default='image')

parser.add_argument("--finalout_type", choices=['image', 'rgba', 'fgbg'], default='image')

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

nameStr = 'baseline23' + '_L1Grd_' + str(a.l1_weight_grd) + '_PerGrd_' + str(a.perceptual_weight_grd)

def save_images(fetches, step=None):
    cmap = np.load('../cmap.npy')
    image_dir = os.path.join('./GeneratedData/', a.dataset, nameStr, 'image')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["outputs"]:
            filename = name + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)

        filesets.append(fileset)
    return filesets


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    cmap = np.load('../cmap.npy')

    output_dir = os.path.join(a.dataset, nameStr, 'aer')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        checkpoint_dir = os.path.join(a.dataset, nameStr, 'aer')
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.dataset=='CVUSA':
        from load_data.load_data_cvusa import load_examples
    elif a.dataset=='CVACT':
        from load_data.load_data_cvact import load_examples
    elif a.dataset=='CVACThalf':
        from load_data.load_data_cvact_half import load_examples
    elif a.dataset=='OP':
        from load_data.load_data_op import load_examples

    examples = load_examples(a.mode, a.batch_size)
    print("examples count = %d" % examples.count)

    inputs = examples.aer
    targets = examples.pano
    ref_images = examples.tanpolar

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(inputs, targets, ref_images, a)

    inputs = deprocess(inputs)
    targets = deprocess(targets)
    outputs = deprocess(model.outputs)
    converted_generator_inputs = deprocess(model.generator_inputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),

        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("gen_loss_perceptual", model.gen_loss_perceptual)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint_dir = os.path.join(a.dataset, nameStr, 'aer')
            checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                # index_path = append_index(filesets)
            # print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1
                    fetches["gen_loss_perceptual"] = model.gen_loss_perceptual

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)
                # height = sess.run(model.estimated_height, options=options, run_metadata=run_metadata)
                # for b in range(0, a.batch_size):
                #     img = cmap[height[b].squeeze()]
                #     img = Image.fromarray(img)
                #     img.save(str(b)+'height.png')


                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])
                    print("gen_loss_perceptual", results["gen_loss_perceptual"])

                if should(examples.steps_per_epoch):
                # if should(50):
                    print("saving model")
                    saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


main()
