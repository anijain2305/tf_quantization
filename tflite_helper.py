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

########################
# TFLite convertor class
########################
class TFLiteConvertor(object):
    def __init__(self, saved_model_dir, base_path, model_name):
        try:
            self.convertor = tf.lite.TFLiteConverter.from_keras_model(saved_model_dir)
        except:
            self.convertor = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        self.base_path = base_path
        self.model_name = model_name
    
    def fp32(self):
        model_name = self.model_name + "_fp32.tflite"
        tflite_model_path = pathlib.Path(self.base_path)/model_name
        tflite_model_path.write_bytes(self.convertor.convert())

    def weight_int8_act_fp32(self):
        model_name = self.model_name + "_weight_int8_act_fp32.tflite"
        tflite_model_path = pathlib.Path(self.base_path)/model_name

        self.convertor.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_model_path.write_bytes(self.convertor.convert())

    def full_integer_except_io(self, representative_data_gen):
        model_name = self.model_name + "_full_integer_except_io.tflite"
        tflite_model_path = pathlib.Path(self.base_path)/model_name

        self.convertor.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        self.convertor.representative_dataset = representative_data_gen
        tflite_model_path.write_bytes(self.convertor.convert())
