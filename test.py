# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:45:40 2019

@author: Akash Sabir
"""
from keras.models import load_model
m = load_model('facenet/facenet_keras.h5')
m.summary()