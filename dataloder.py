from gensim.models.keyedvectors import KeyedVectors
from konlpy.tag import Mecab

import time
import numpy as np

import os
import sys
import urllib.request
import datetime
import pickle
import json

def embedding_load():
    en_model = KeyedVectors.load_word2vec_format('./fasttext/wiki.en.vec')
    ko_model = KeyedVectors.load_word2vec_format('./fasttext/wiki.ko.vec')

    return en_model, ko_model



if __name__ == '__main__':
    m = Mecab()