import tensorflow as tf

import numpy as np

import time
import os

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Use at decoder
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=False)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, fast_text_embedding, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, fast_text_embedding, d_model)
        k = self.wk(k)  
        v = self.wv(v)  

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, fast_text_embedding_q, depth)
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size)  


        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        print("Scaled_attention Shape : ", scaled_attention.shape)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  
        
        print("Scaled_attention Shape : ", scaled_attention.shape)
        
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  
        print("Concat attention Shape :", concat_attention.shape)
        output = self.dense(concat_attention) 
        output = tf.reduce_mean(output, axis=-1, keepdims=True)
        print(output.shape)

        return output, attention_weights
    
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu', use_bias=False),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model, use_bias=False)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(1, dff)
        
        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        
        print(x.shape)
        
        attn_output, _ = self.mha(x, x, x, mask) # (batch, fasttext_embedding_dim, d_model)
        attn_output = self.dropout_1(attn_output, training=training)
        out_1 = self.layernorm_1(x + attn_output)
       
        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output, training=training)
        out_2 = self.layernorm_2(out_1 + ffn_output)
    
        return out_2
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size=0,
               maximum_position_encoding=0, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        
        x = tf.expand_dims(x, -1)
        print("Model Input shape", x.shape)
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  