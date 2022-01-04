import collections
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math

# Examples = collections.namedtuple("Examples", "paths, aer, pano, mask, count, steps_per_epoch")
Examples = collections.namedtuple("Examples", "paths, aer, pano, tanpolar, count, steps_per_epoch")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def load_examples(mode='train', batch_size=2):

    if mode=='train':
        file_list = '../../../Data/OP/splits/train_split.txt'
    else:
        file_list = '../../../Data/OP/splits/test_split.txt'
    img_root = '../../../Data/OP/'

    data_list = []

    with open(file_list, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(',')
            if mode == 'train' and items[0].replace('_nadir', '') != items[2]:
                continue
            else:
                data_list.append([img_root + 'aerial/' + items[0],
                                  img_root + 'panorama/' + items[1].replace('\n', ''),
                                  # img_root + 'refinenetSeman/aerial/' + items[0],
                                  # img_root + 'refinenetSeman/panorama_visualize/' + items[1].replace('\n', ''),
                                  img_root + 'tanpolar/' + items[0],
                                  items[1].replace('\n', '')])

    aer_list = [item[0] for item in data_list]
    pano_list = [item[1] for item in data_list]
    # mask_list = [item[2] for item in data_list]
    tanpolar_list = [item[2] for item in data_list]
    # polar_list = [item[4] for item in data_list]

    aer_queue = tf.train.string_input_producer(aer_list, shuffle=mode=='train', seed=2020)
    pano_queue = tf.train.string_input_producer(pano_list, shuffle=mode=='train', seed=2020)
    # mask_queue = tf.train.string_input_producer(mask_list, shuffle=mode=='train', seed=2020)
    tanpolar_queue = tf.train.string_input_producer(tanpolar_list, shuffle=mode == 'train', seed=2020)
    # polar_queue = tf.train.string_input_producer(polar_list, shuffle=mode == 'train', seed=2020)

    reader = tf.WholeFileReader()
    aer_paths, aer_contents = reader.read(aer_queue)
    pano_paths, pano_contents = reader.read(pano_queue)
    # mask_paths, mask_contents = reader.read(mask_queue)
    tanpolar_paths, tanpolar_contents = reader.read(tanpolar_queue)
    # polar_paths, polar_contents = reader.read(polar_queue)

    aer = tf.image.decode_jpeg(aer_contents)
    panos = tf.image.decode_jpeg(pano_contents)
    # mask = tf.image.decode_png(mask_contents)
    tanpolar = tf.image.decode_png(tanpolar_contents)
    # polar = tf.image.decode_png(polar_contents)

    aer = tf.image.convert_image_dtype(aer, tf.float32)
    panos = tf.image.convert_image_dtype(panos, tf.float32)
    tanpolar = tf.image.convert_image_dtype(tanpolar, tf.float32)
    # mask = tf.image.convert_image_dtype(mask, tf.float32)
    # polar = tf.image.convert_image_dtype(polar, tf.float32)

    aer = preprocess(aer)
    panos = preprocess(panos)
    tanpolar = preprocess(tanpolar)
    # mask = preprocess(mask)
    # polar = preprocess(polar)

    aer.set_shape([None, None, 3])
    panos.set_shape([None, None, 3])
    # mask.set_shape([None, None, 3])
    tanpolar.set_shape([None, None, 3])
    # polar.set_shape([None, None, 3])

    aer = tf.image.resize_images(aer, [256, 256], method=tf.image.ResizeMethod.AREA)
    panos = tf.image.resize_images(panos, [128, 512], method=tf.image.ResizeMethod.AREA)
    # mask = tf.image.resize_images(mask, [128, 512], method=tf.image.ResizeMethod.AREA)
    # mask = tf.cast(tf.image.resize_images(mask, [128, 512], method=tf.image.ResizeMethod.AREA), tf.int32)
    # mask = 0.9 * tf.one_hot(tf.squeeze(mask, axis=-1), depth=4)
    tanpolar = tf.image.resize_images(tanpolar, [128, 512], method=tf.image.ResizeMethod.AREA)
    # polar = tf.image.resize_images(polar, [128, 512], method=tf.image.ResizeMethod.AREA)

    aer_batch, panos_batch, pano_paths_batch, tanpolar_batch = \
        tf.train.batch([aer, panos, pano_paths, tanpolar], batch_size=batch_size)
    # aer_batch, panos_batch, mask_batch, aer_paths_batch = \
    #     tf.train.batch([aer, panos, mask, aer_paths], batch_size=batch_size)

    steps_per_epoch = int(math.ceil(len(data_list) / batch_size))

    return Examples(
            paths=pano_paths_batch,
            aer=aer_batch,
            pano=panos_batch,
            # mask=mask_batch,
            tanpolar=tanpolar_batch,
            # polar=polar_batch,
            count=len(data_list),
            steps_per_epoch=steps_per_epoch,
        )


