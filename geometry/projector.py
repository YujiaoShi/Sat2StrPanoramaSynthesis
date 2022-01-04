import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from MIP import tf_shape


def over_composite(rgbas):
  """Combines a list of RGBA images using the over operation.

  Combines RGBA images from back to front with the over operation.
  The alpha image of the first image is ignored and assumed to be 1.0.

  Args:
    rgbas: A list of [batch, H, W, 4] RGBA images, combined from back to front.
  Returns:
    Composited RGB image.
  """
  for i in range(len(rgbas)):
    rgb = rgbas[i][:, :, :, 0:3]
    alpha = rgbas[i][:, :, :, 3:]
    if i == 0:
      output = rgb
    else:
      rgb_by_alpha = rgb * alpha
      output = rgb_by_alpha + output * (1.0 - alpha)
  return output


def mpi_render_grd_view(batch_rgbas, share_alpha=True):

  batch, height, width, channel = batch_rgbas.get_shape().as_list()

  if share_alpha:
    num_mpi_planes = int(channel/4)
    rgba_layers = tf.reshape(batch_rgbas, [-1, height, width, num_mpi_planes, 4])
    rgb = rgba_layers[..., :3]
    alpha = tf.expand_dims(rgba_layers[..., -1], axis=-1)
  else:
    num_mpi_planes = int(channel / 5)
    rgba_layers = tf.reshape(batch_rgbas, [-1, height, width, num_mpi_planes, 5])
    rgb = rgba_layers[..., :3]
    alpha = tf.expand_dims(rgba_layers[..., 4], axis=-1)

  alpha = (alpha + 1.)/2.
  rgba_layers = tf.transpose(tf.concat([rgb, alpha], axis=-1), [3, 0, 1, 2, 4])

  rgba_list = []
  for i in range(int(num_mpi_planes)):
    rgba_list.append(rgba_layers[i])

  synthesis_image = over_composite(rgba_list)
  # shape = [batch, height, width, 3]

  return synthesis_image


def mpi_render_aer_view(batch_rgbas, share_alpha=True):
  batch, height, width, channel = batch_rgbas.get_shape().as_list()

  if share_alpha:
    num_mpi_planes = int(channel / 4)
    rgba_layers = tf.reshape(batch_rgbas, [-1, height, width, num_mpi_planes, 4])
    rgb = rgba_layers[..., :3]
    alpha = tf.expand_dims(rgba_layers[..., -1], axis=-1)
  else:
    num_mpi_planes = int(channel / 5)
    rgba_layers = tf.reshape(batch_rgbas, [-1, height, width, num_mpi_planes, 5])
    rgb = rgba_layers[..., :3]
    alpha = tf.expand_dims(rgba_layers[..., -1], axis=-1)
  alpha = (alpha + 1.) / 2.
  rgba_layers = tf.transpose(tf.concat([rgb, alpha], axis=-1), [1, 0, 2, 3, 4])
  # shape = [height, batch, width, num_mpi_planes, 4]

  rgba_list = []
  for i in range(int(height)):
    rgba_list.append(rgba_layers[i])

  rgba_list = rgba_list[::-1][:int(height//2)]

  synthesis_image = over_composite(rgba_list)
  # shape = [batch, width, num_mpi_planes, 3]

  return synthesis_image


def rtheta2uv(athetaimage, aer_size):
  '''
  :param athetaimage: shape = [batch, width, PlaneNum, 3]  width-->theta PlaneNum-->radius
  :param aer_size:
  :return:
  '''
  batch, width, PlaneNum, channel = tf_shape(athetaimage, 4)
  i = np.arange(aer_size)
  j = np.arange(aer_size)
  jj, ii = np.meshgrid(j, i)

  center = aer_size / 2 - 0.5
  theta = np.arctan(-(jj - center) / (ii - center))
  theta[np.where(ii < center)] += np.pi
  theta[np.where((ii >= center) & (jj >= center))] += 2 * np.pi
  theta = theta/(2 * np.pi)*width

  RadiusByPixel = np.sqrt((ii - center) ** 2 + (jj - center) ** 2)
  RadiusByPixel = (1-RadiusByPixel/aer_size*2)*PlaneNum

  uv = np.stack([RadiusByPixel, theta], axis=-1)
  uv = uv.astype(np.float32)
  warp = tf.stack([uv] * batch, axis=0)

  sampler_output = tf.contrib.resampler.resampler(athetaimage, warp)
  # shape = [batch, aer_size, aer_size, 3]

  return sampler_output







