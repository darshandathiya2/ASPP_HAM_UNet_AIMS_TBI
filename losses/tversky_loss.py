
import tensorflow as tf
from tensorflow.keras import backend as K

def tversky_index(y_true,
                  y_pred,
                  alpha=0.7,
                  beta=0.3,
                  smooth=1e-6):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))

    return (tp + smooth) / (
        tp + alpha * fp + beta * fn + smooth
    )

def tversky_loss(y_true, y_pred):
    return 1 - tversky_index(y_true, y_pred)
