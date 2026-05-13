
import tensorflow as tf

def focal_loss(alpha=0.25, gamma=2.0):

    def loss(y_true, y_pred):

        epsilon = 1e-7
        y_pred = tf.clip_by_value(
            y_pred,
            epsilon,
            1.0 - epsilon
        )

        cross_entropy = -y_true * tf.math.log(y_pred)

        weight = alpha * tf.pow(1 - y_pred, gamma)

        return tf.reduce_mean(weight * cross_entropy)

    return loss
