import collections
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import scipy.io as sio
import os

# Examples = collections.namedtuple("Examples", "paths, aer, pano, mask, count, steps_per_epoch, tanpolar, polar")
Examples = collections.namedtuple("Examples", "paths, aer, pano, count, steps_per_epoch, tanpolar, polar")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def load_examples(mode='train', batch_size=2):

    # allDataList = '../OriNet_CVACT/CVACT_orientations/ACT_data.mat'
    img_root = '../../../Data/CVACT/'
    allDataList = os.path.join(img_root, 'ACT_data.mat')

    exist_aer_list = os.listdir(img_root + 'satview_correct')
    exist_grd_list = os.listdir(img_root + 'streetview')

    __cur_allid = 0  # for training

    # load the mat
    anuData = sio.loadmat(allDataList)

    data_list = []
    for i in range(0, len(anuData['panoIds'])):
        # grd_id_align = img_root + 'streetview/' + anuData['panoIds'][i] + '_grdView.png'
        # sat_id_ori = img_root + 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.png'
        grd_id_align = anuData['panoIds'][i] + '_grdView.png'
        sat_id_ori = anuData['panoIds'][i] + '_satView_polish.png'
        data_list.append([grd_id_align, sat_id_ori])

    if mode=='train':
        training_inds = anuData['trainSet']['trainInd'][0][0] - 1
        trainNum = len(training_inds)
        trainList = []
        for k in range(trainNum):
            trainList.append(data_list[training_inds[k][0]])
        pano_list = [img_root + 'streetview/' + item[0] for item in trainList if item[0] in exist_grd_list and item[1] in exist_aer_list]
        aer_list = [img_root + 'satview_correct/' + item[1] for item in trainList if item[0] in exist_grd_list and item[1] in exist_aer_list]
        # pano_seman_list = [img_root + 'streetseman_visualize/' + item[0] for item in trainList if
        #              item[0] in exist_grd_list and item[1] in exist_aer_list]
        tanpolar_list = [img_root + 'a2g_correct/' + item[1] for item in trainList if
                    item[0] in exist_grd_list and item[1] in exist_aer_list]
        polar_list = [img_root + 'polarmap/' + item[1] for item in trainList if
                    item[0] in exist_grd_list and item[1] in exist_aer_list]


    else:

        val_inds = anuData['valSet']['valInd'][0][0] - 1
        valNum = len(val_inds)
        valList = []
        for k in range(valNum):
            valList.append(data_list[val_inds[k][0]])
        pano_list = [img_root + 'streetview/' + item[0] for item in valList if item[0] in exist_grd_list and item[1] in exist_aer_list]
        aer_list = [img_root + 'satview_polish/' + item[1] for item in valList if item[0] in exist_grd_list and item[1] in exist_aer_list]
        # pano_seman_list = [img_root + 'streetseman_visualize/' + item[0] for item in valList if
        #              item[0] in exist_grd_list and item[1] in exist_aer_list]
        # aer_seman_list = [img_root + 'satseman/' + item[1] for item in valList if
        #             item[0] in exist_grd_list and item[1] in exist_aer_list]
        tanpolar_list = [img_root + 'a2g_origin/' + item[1] for item in valList if
                    item[0] in exist_grd_list and item[1] in exist_aer_list]
        polar_list = [img_root + 'polarmap/' + item[1] for item in valList if
                    item[0] in exist_grd_list and item[1] in exist_aer_list]

    aer_queue = tf.train.string_input_producer(aer_list, shuffle=mode == 'train', seed=2020)
    pano_queue = tf.train.string_input_producer(pano_list, shuffle=mode == 'train', seed=2020)
    # pano_seman_queue = tf.train.string_input_producer(pano_seman_list, shuffle=mode == 'train', seed=2020)
    tanpolar_queue = tf.train.string_input_producer(tanpolar_list, shuffle=mode == 'train', seed=2020)
    polar_queue = tf.train.string_input_producer(polar_list, shuffle=mode == 'train', seed=2020)

    reader = tf.WholeFileReader()
    aer_paths, aer_contents = reader.read(aer_queue)
    pano_paths, pano_contents = reader.read(pano_queue)
    # pano_seman_paths, pano_seman_contents = reader.read(pano_seman_queue)
    tanpolar_paths, tanpolar_contents = reader.read(tanpolar_queue)
    polar_paths, polar_contents = reader.read(polar_queue)

    aer = tf.image.decode_png(aer_contents)
    panos = tf.image.decode_png(pano_contents)
    # panos_seman = tf.image.decode_png(pano_seman_contents)
    tanpolar = tf.image.decode_png(tanpolar_contents)
    polar = tf.image.decode_png(polar_contents)

    aer = tf.image.convert_image_dtype(aer, tf.float32)
    panos = tf.image.convert_image_dtype(panos, tf.float32)
    # panos_seman = tf.image.convert_image_dtype(panos_seman, tf.float32)
    tanpolar = tf.image.convert_image_dtype(tanpolar, tf.float32)
    polar = tf.image.convert_image_dtype(polar, tf.float32)

    aer = preprocess(aer)
    panos = preprocess(panos)
    # panos_seman = preprocess(panos_seman)
    tanpolar = preprocess(tanpolar)
    polar = preprocess(polar)

    aer.set_shape([None, None, 3])
    panos.set_shape([None, None, 3])
    # panos_seman.set_shape([None, None, 3])
    tanpolar.set_shape([None, None, 3])
    polar.set_shape([None, None, 3])

    aer = tf.image.resize_images(aer, [256, 256], method=tf.image.ResizeMethod.AREA)
    panos = tf.image.resize_images(panos, [128, 512], method=tf.image.ResizeMethod.AREA)
    # panos_seman = tf.image.resize_images(panos_seman, [128, 512], method=tf.image.ResizeMethod.AREA)
    # panos_seman = tf.cast(tf.image.resize_images(panos_seman, [128, 512], method=tf.image.ResizeMethod.AREA), tf.int32)
    # panos_seman = tf.one_hot(tf.squeeze(panos_seman, axis=-1), depth=4)
    tanpolar = tf.image.resize_images(tanpolar, [128, 512], method=tf.image.ResizeMethod.AREA)
    polar = tf.image.resize_images(polar, [128, 512], method=tf.image.ResizeMethod.AREA)

    aer_batch, panos_batch, grd_paths_batch, tanpolar_batch, polar_batch = \
        tf.train.batch([aer, panos, pano_paths, tanpolar, polar], batch_size=batch_size)

    steps_per_epoch = int(math.ceil(len(pano_list) / batch_size))

    return Examples(
        paths=grd_paths_batch,
        aer=aer_batch,
        pano=panos_batch,
        # mask=panos_seman_batch,
        tanpolar=tanpolar_batch,
        polar = polar_batch,
        count=len(pano_list),
        steps_per_epoch=steps_per_epoch,
    )
