import tensorflow as tf

from gensim.models.keyedvectors import KeyedVectors
from konlpy.tag import Mecab

# from googletrans import Translator
from models.transformer import *

import time
import numpy as np

import os
import sys
import urllib.request
import requests
import datetime
import pickle
import json


def load_vocab():
    with open('./data/ko_noun_dict(vocab_size-1250).pkl', "rb") as f:
        ko_dict = pickle.load(f)

    with open('./data/en_noun_dict(vocab_size-1250).pkl', "rb") as f:
        en_dict = pickle.load(f)

    return ko_dict, en_dict


def model_build_train(ko_data, en_data):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(300, bias_initializer='zero'))


    es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    model.fit(ko_data, en_data, epochs=50000, batch_size=64, callbacks=[es])

    return model

if __name__ == '__main__':
    ko_dict, en_dict = load_vocab()

    ko_data = np.array(list(ko_dict.values()))
    en_data = np.array(list(en_dict.values()))

    model = model_build_train(ko_data, en_data)
