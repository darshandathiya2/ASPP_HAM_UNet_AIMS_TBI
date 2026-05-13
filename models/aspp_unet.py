
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def ASPP(inputs, filters):

    conv1 = Conv2D(filters, 1, padding='same',
                   dilation_rate=1,
                   activation='relu')(inputs)

    conv6 = Conv2D(filters, 3, padding='same',
                   dilation_rate=6,
                   activation='relu')(inputs)

    conv12 = Conv2D(filters, 3, padding='same',
                    dilation_rate=12,
                    activation='relu')(inputs)

    conv18 = Conv2D(filters, 3, padding='same',
                    dilation_rate=18,
                    activation='relu')(inputs)

    concat = Concatenate()([
        conv1,
        conv6,
        conv12,
        conv18
    ])

    output = Conv2D(filters, 1,
                    padding='same',
                    activation='relu')(concat)

    return output
