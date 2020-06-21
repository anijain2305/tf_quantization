try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
tf.enable_v2_behavior()

# from tensorflow import keras
import numpy as np
import glob
import random

from tflite_inference import TFLiteExecutor
from accuracy_measurement import AccuracyAggregator

def calculate_accuracy(test_dataset, tflite_model_path, im_height, im_width, preprocess_f,
        postprocess_f, num_images):
	tflite_executor = TFLiteExecutor(tflite_model_path)
	accuracy_aggregator = AccuracyAggregator()
	total = 0
	for image_instance in test_dataset:
	    total = total + 1
	    
	    # Preprocess the image
	    preprocessed_image = preprocess_f(image_instance, im_height, im_width)
	    
	    tflite_output = tflite_executor.run(preprocessed_image)
	    tflite_output = postprocess_f(tflite_output)
	    accuracy_aggregator.update(image_instance, tflite_output)
	    if total == num_images:
	        return (accuracy_aggregator.report())
	return (accuracy_aggregator.report())


def get_datasets():
    imagenet_path = '/home/ubuntu/imagenet/val/'
    all_class_path = sorted(glob.glob(imagenet_path+'*'))
    
    images = list()
    for cur_class in all_class_path:
        all_image = glob.glob(cur_class+'/*')
        images.extend(all_image)
    
    random.seed(0)
    random.shuffle(images)
    
    
    calibration_dataset = images[0:1000]
    test_dataset = images[1000:]
    return (calibration_dataset, test_dataset)
