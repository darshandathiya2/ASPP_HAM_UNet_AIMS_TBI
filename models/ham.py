
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def channel_attention(inputs):

    avg_pool = GlobalAveragePooling2D()(inputs)
    max_pool = GlobalMaxPooling2D()(inputs)

    avg_pool = Reshape((1,1,inputs.shape[-1]))(avg_pool)
    max_pool = Reshape((1,1,inputs.shape[-1]))(max_pool)

    concat = Add()([avg_pool, max_pool])

    dense = Conv2D(inputs.shape[-1], 1, activation='sigmoid')(concat)

    return Multiply()([inputs, dense])

def spatial_attention(inputs):

    avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    attention = Conv2D(1, 7,
                       padding='same',
                       activation='sigmoid')(concat)

    return Multiply()([inputs, attention])

def HAM(inputs):

    x = channel_attention(inputs)
    x = spatial_attention(x)

    return x
