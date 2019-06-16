import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(keras.layers.Layer):
    """
        Maps MNIST digits to a triplet (z_mean, z_log_var, z).
        """
    def __init__(self, latent_dim=2, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv_1 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv_2 = keras.layers.Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')
        self.conv_3 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv_4 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(32, 'relu')
        self.dense_mean = keras.layers.Dense(latent_dim)
        self.dense_log_var = keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(keras.layers.Layer):
    """
        Converts z, the encoded digit vector, back into a readable digit.
        """
    def __init__(self, shape_before_flattening, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense = keras.layers.Dense(np.prod(shape_before_flattening), activation='relu')
        self.reshape = keras.layers.Reshape(target_shape=shape_before_flattening)
        self.deconv = keras.layers.Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation='relu')
        self.conv = keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.deconv(x)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    input_shape = (28, 28, 1)
    shape_before_flattening = (14, 14, 1)
    inputs = keras.Input(input_shape)
    z_mean, z_log_var, z = Encoder()(inputs)
    reconstructed = Decoder(shape_before_flattening)(z)
    model = keras.Model(inputs, reconstructed)
    model.summary()
    print()