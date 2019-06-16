import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


class MSELoss(LossFunctionWrapper):
    def __init__(self, name='mse_loss'):
        super(MSELoss, self).__init__(mse_loss, name=name)


def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(y_true, y_pred), axis=[1, 2, 3]))
