print("Testing environment setup:")
import os
print(os.getenv('TF_ENABLE_ONEDNN_OPTS'))  # Ar trebui să afișeze '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = toate mesajele, 1 = ignoră info, 2 = ignoră warnings, 3 = doar erorile
import tensorflow as tf
print('TensorFlow:', tf.__version__)

import keras
import cv2
import sklearn
import dlib
import numpy as np
import matplotlib.pyplot as plt

print("All libraries are working correctly!")
