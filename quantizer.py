try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
tf.enable_v2_behavior()
import tensorflow_hub as hub

# from tensorflow import keras
import numpy as np
import pathlib
from util import get_datasets, calculate_accuracy
from processing import preprocess, postprocess
from tflite_helper import TFLiteConvertor

##################
# Prepare datasets
##################

cv_dataset, test_dataset = get_datasets()
num_images_in_cv_dataset = 100

# For 224 x 224
preprocess_cv_dataset_224 = list()
for i in range(0, num_images_in_cv_dataset):
    preprocess_cv_dataset_224.append(preprocess(cv_dataset[i], 224, 224))
 
# For 299 x 299
preprocess_cv_dataset_299 = list()
for i in range(0, num_images_in_cv_dataset):
    preprocess_cv_dataset_299.append(preprocess(cv_dataset[i], 299, 299))
 
def representative_data_gen_224():
    for input_image in preprocess_cv_dataset_224:
        yield [input_image]

def representative_data_gen_299():
    for input_image in preprocess_cv_dataset_299:
        yield [input_image]


##############################
# Original FP32 TF/Keras model
##############################
tf_hub_links = {
    "resnet_50"             : "https://tfhub.dev/tensorflow/resnet_50/classification/1",
    "resnet_v2_50"          : "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/4",
    "mobilenet_v1"          : "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4",
    "mobilenet_v2"          : "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4",
    "inception_v1"          : "https://tfhub.dev/google/imagenet/inception_v1/classification/4",
    "inception_v2"          : "https://tfhub.dev/google/imagenet/inception_v2/classification/4",
    "inception_v3"          : "https://tfhub.dev/google/imagenet/inception_v3/classification/4",
    "inception_v3_preview"  : "https://tfhub.dev/google/tf2-preview/inception_v3/classification/4",
    "mobilenet_v2_preview"  : "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4",
    # "efficientnet_b0"       : "https://tfhub.dev/tensorflow/efficientnet/b0/classification/1",
}

def quantize(model_name, out_path):
    keras_model = tf.keras.Sequential([
        hub.KerasLayer(tf_hub_links[model_name], output_shape=[1001])
    ])

    if "inception_v3" in model_name:
        keras_model._set_inputs(preprocess_cv_dataset_299[0])  # Batch input shape.
    else:
        keras_model._set_inputs(preprocess_cv_dataset_224[0])  # Batch input shape.
    
    tflite_convertor = TFLiteConvertor(\
            saved_model_dir=keras_model,
            base_path=out_path,
            model_name=model_name)
    
    tflite_convertor.fp32()
    tflite_convertor.weight_int8_act_fp32()
    if "inception_v3" in model_name:
        tflite_convertor.full_integer_except_io(representative_data_gen_299)
    else:
        tflite_convertor.full_integer_except_io(representative_data_gen_224)

    im_height = 299 if "inception_v3" in model_name else 224
    im_width = im_height

    tflite_model_path = out_path + "/" + model_name + "_fp32.tflite"
    top1, top5 = calculate_accuracy(test_dataset, tflite_model_path,
                                    im_height, im_width,
                                    preprocess,
                                    postprocess[model_name], 10)
    print("{:15} {:20} {:10} {:10}".format(model_name, "fp32", top1, top5))

    tflite_model_path = out_path + "/" + model_name + "_full_integer_except_io.tflite"
    top1, top5 = calculate_accuracy(test_dataset, tflite_model_path,
                                    im_height, im_width,
                                    preprocess,
                                    postprocess[model_name], 10)
    print("{:15} {:20} {:10} {:10}".format(model_name, "full_integer", top1, top5))

out_path = "/tmp"
for model in tf_hub_links.keys():
    quantize(model, out_path)
