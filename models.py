import tensorflow.keras as keras
from layers import Encoder, Decoder


class VariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""
    def __init__(self, input_shape=(28, 28, 1), latent_dim=16, name='autoencoder', **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        shape_before_flatten = (input_shape[0]//2, input_shape[1]//2, 1)
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(shape_before_flatten)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed

