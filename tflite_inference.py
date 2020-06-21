""" TFlite inference functions """
try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
tf.enable_v2_behavior()

import numpy as np

class TFLiteExecutor(object):
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()

    def run(self, input_data):
        def convert_to_list(x):
            if not isinstance(x, list):
                x = [x]
            return x

        """ Generic function to execute TFLite """
        input_data = convert_to_list(input_data)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # set input
        assert len(input_data) == len(input_details)
        for i in range(len(input_details)):
            self.interpreter.set_tensor(input_details[i]['index'], input_data[i])

        self.interpreter.invoke()

        # get output
        tflite_output = list()
        for i in range(len(output_details)):
            tflite_output.append(self.interpreter.get_tensor(output_details[i]['index']))

        return tflite_output
