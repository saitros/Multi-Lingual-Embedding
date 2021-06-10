import tensorflow as tf
import numpy as np
import time
import os

class AutoEncoder(tf.keras.models.Model):
    def __init__(self, latent_dim):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_dim, activation='relu', use_bias=False),
#             tf.keras.layers.Dense(self.latent_dim-50, activation='relu', use_bias=False),
        ])
        self.decoder = tf.keras.Sequential([
#             tf.keras.layers.Dense(latent_dim, activation = 'relu'),
            tf.keras.layers.Dense(300, activation = 'sigmoid'),
        ])
        
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    