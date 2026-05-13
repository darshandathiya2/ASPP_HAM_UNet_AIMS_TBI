
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def conv_block(x, filters):

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def build_unet(input_shape=(256, 256, 1)):

    inputs = Input(input_shape)

    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D()(c3)

    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D()(c4)

    bn = conv_block(p4, 1024)

    u1 = UpSampling2D()(bn)
    u1 = Concatenate()([u1, c4])
    c5 = conv_block(u1, 512)

    u2 = UpSampling2D()(c5)
    u2 = Concatenate()([u2, c3])
    c6 = conv_block(u2, 256)

    u3 = UpSampling2D()(c6)
    u3 = Concatenate()([u3, c2])
    c7 = conv_block(u3, 128)

    u4 = UpSampling2D()(c7)
    u4 = Concatenate()([u4, c1])
    c8 = conv_block(u4, 64)

    outputs = Conv2D(1, 1, activation="sigmoid")(c8)

    return Model(inputs, outputs)
