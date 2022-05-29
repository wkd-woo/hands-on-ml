import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import tensorflowjs as tfjs

vgg16 = keras.applications.vgg16.VGG16()
tfjs.converters.save_keras_model(vgg16, './public/tfjs-models/VGG16')

mobilenetv2 = keras.applications.MobileNetV2()
tfjs.converters.save_keras_model(mobilenetv2, './public/tfjs-models/Mobilenetv2')