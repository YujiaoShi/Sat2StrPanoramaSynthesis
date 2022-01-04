import collections
import os.path

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math

# Examples = collections.namedtuple("Examples", "paths, aer, pano, mask, count, steps_per_epoch")
Examples = collections.namedtuple("Examples", "paths, aer, pano, tanpolar, polar, count, steps_per_epoch")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def load_examples(mode='train', batch_size=2):

    img_root = '../../../Data/CVUSA/'

    if mode=='train':
        file_list = os.path.join(img_root, 'splits/train-19zl.csv')
    elif mode=='test':
        file_list = os.path.join(img_root, 'splits/val-19zl.csv')

    data_list = []

    with open(file_list, 'r') as f:
        for line in f:
            data = line.split(',')
            # data_list.append([img_root + data[0], img_root + data[1], img_root + data[2][:-1]])
            data_list.append([img_root + data[0], img_root + data[1],
                              # img_root + data[2][:-1].replace('annotations', 'annotations_visualize'),
                              img_root + data[0].replace('bingmap/19', 'a2g').replace('jpg', 'png'),
                              img_root + data[0].replace('bing', 'polar').replace('jpg', 'png')])

    aer_list = [item[0] for item in data_list]
    pano_list = [item[1] for item in data_list]
    # mask_list = [item[2] for item in data_list]
    tanpolar_list = [item[2] for item in data_list]
    polar_list = [item[3] for item in data_list]

    aer_queue = tf.train.string_input_producer(aer_list, shuffle=mode=='train', seed=2020)
    pano_queue = tf.train.string_input_producer(pano_list, shuffle=mode=='train', seed=2020)
    tanpolar_queue = tf.train.string_input_producer(tanpolar_list, shuffle=mode == 'train', seed=2020)
    polar_queue = tf.train.string_input_producer(polar_list, shuffle=mode == 'train', seed=2020)

    # aer_queue = tf.data.Dataset.from_tensor_slices(aer_list)
    # pano_queue = tf.data.Dataset.from_tensor_slices(pano_list)
    # tanpolar_queue = tf.data.Dataset.from_tensor_slices(tanpolar_list)
    # polar_queue = tf.data.Dataset.from_tensor_slices(polar_list)
    # if mode=='train':
    #     buffer_size = len(data_list)
    #     aer_queue = aer_queue.shuffle(buffer_size, seed=2020)
    #     pano_queue = pano_queue.shuffle(buffer_size, seed=2020)
    #     tanpolar_queue = tanpolar_queue.shuffle(buffer_size, seed=2020)
    #     polar_queue = polar_queue.shuffle(buffer_size, seed=2020)

    reader = tf.WholeFileReader()
    aer_paths, aer_contents = reader.read(aer_queue)
    pano_paths, pano_contents = reader.read(pano_queue)
    # mask_paths, mask_contents = reader.read(mask_queue)
    tanpolar_paths, tanpolar_contents = reader.read(tanpolar_queue)
    polar_paths, polar_contents = reader.read(polar_queue)

    aer = tf.image.decode_jpeg(aer_contents)
    panos = tf.image.decode_jpeg(pano_contents)
    # mask = tf.image.decode_png(mask_contents)
    tanpolar = tf.image.decode_png(tanpolar_contents)
    polar = tf.image.decode_png(polar_contents)

    aer = tf.image.convert_image_dtype(aer, tf.float32)
    panos = tf.image.convert_image_dtype(panos, tf.float32)
    # mask = tf.image.convert_image_dtype(mask, tf.float32)
    tanpolar = tf.image.convert_image_dtype(tanpolar, tf.float32)
    polar = tf.image.convert_image_dtype(polar, tf.float32)

    aer = preprocess(aer)
    panos = preprocess(panos)
    # mask = preprocess(mask)
    tanpolar = preprocess(tanpolar)
    polar = preprocess(polar)

    aer.set_shape([None, None, 3])
    panos.set_shape([None, None, 3])
    # mask.set_shape([None, None, 3])
    tanpolar.set_shape([None, None, 3])
    polar.set_shape([None, None, 3])

    aer = tf.image.resize_images(aer, [256, 256], method=tf.image.ResizeMethod.AREA)
    panos = tf.image.resize_images(panos, [128, 512], method=tf.image.ResizeMethod.AREA)
    # mask = tf.image.resize_images(mask, [128, 512], method=tf.image.ResizeMethod.AREA)
    # mask = tf.cast(tf.image.resize_images(mask, [128, 512], method=tf.image.ResizeMethod.AREA), tf.int32)
    # mask = 0.9 * tf.one_hot(tf.squeeze(mask, axis=-1), depth=4)
    tanpolar = tf.image.resize_images(tanpolar, [128, 512], method=tf.image.ResizeMethod.AREA)
    polar = tf.image.resize_images(polar, [128, 512], method=tf.image.ResizeMethod.AREA)

    # aer_batch, panos_batch, mask_batch, aer_paths_batch, tanpolar_batch, polar_batch = \
    #     tf.train.batch([aer, panos, mask, aer_paths, tanpolar, polar], batch_size=batch_size)
    aer_batch, panos_batch, aer_paths_batch, tanpolar_batch, polar_batch = \
        tf.train.batch([aer, panos, aer_paths, tanpolar, polar], batch_size=batch_size)

    steps_per_epoch = int(math.ceil(len(data_list) / batch_size))

    return Examples(
            paths=aer_paths_batch,
            aer=aer_batch,
            pano=panos_batch,
            # mask=mask_batch,
            tanpolar=tanpolar_batch,
            polar=polar_batch,
            count=len(data_list),
            steps_per_epoch=steps_per_epoch,
        )



