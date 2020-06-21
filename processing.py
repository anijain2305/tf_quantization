""" Preprocess and postprocess functions. """
try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
tf.enable_v2_behavior()

import numpy as np


def preprocess(image_instance,
               height,
               width,
               central_fraction=0.875,
               scope=None,
               central_crop=True,
               use_grayscale=False):
  """Prepare one image for evaluation.
  If height and width are specified it would output an image with that size by
  applying resize_bilinear.
  If central_fraction is specified it would crop the central fraction of the
  input image.
  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
    central_crop: Enable central cropping of images during preprocessing for
      evaluation.
    use_grayscale: Whether to convert the image from RGB to grayscale.
  Returns:
    3-D float Tensor of prepared image.
  """

  image = tf.io.read_file(image_instance)
  image = tf.image.decode_jpeg(image, channels=3)
  with tf.name_scope('eval_image'):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if use_grayscale:
      image = tf.image.rgb_to_grayscale(image)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_crop and central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.compat.v1.image.resize(image, [height, width],
                                        align_corners=False)
      image = tf.image.resize(image, [height, width])
      image = tf.squeeze(image, [0])
    # image = tf.subtract(image, 0.5)
    # image = tf.multiply(image, 2.0)
    image = tf.expand_dims(image, axis=0)
    return image


def postprocess_index0_background(tensor):
    tensor = np.array(tensor)
    predictions = np.squeeze(tensor)
    return predictions


def postprocess_index(tensor):
    tensor = np.array(tensor)
    tensor = tensor[:, :, 1:]
    predictions = np.squeeze(tensor)
    return predictions

postprocess = {
    "resnet_50"             : postprocess_index0_background,
    "resnet_v2_50"          : postprocess_index,
    "mobilenet_v1"          : postprocess_index,
    "mobilenet_v2"          : postprocess_index,
    "mobilenet_v2_preview"  : postprocess_index,
    "inception_v1"          : postprocess_index,
    "inception_v2"          : postprocess_index,
    "inception_v3"          : postprocess_index,
    "inception_v3_preview"  : postprocess_index,
    "mobilenet_v2_preview"  : postprocess_index,
    "efficientnet_b0"       : postprocess_index,
}
