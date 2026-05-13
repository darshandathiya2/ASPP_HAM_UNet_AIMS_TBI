
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def attention_gate(x, g, inter_channel):

    theta_x = Conv2D(inter_channel, 1)(x)
    phi_g = Conv2D(inter_channel, 1)(g)

    add = Add()([theta_x, phi_g])
    act = Activation("relu")(add)

    psi = Conv2D(1, 1)(act)
    psi = Activation("sigmoid")(psi)

    out = Multiply()([x, psi])

    return out
