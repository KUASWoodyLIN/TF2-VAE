import os
import tensorflow as tf
import tensorflow_datasets as tfds
from models import VariationalAutoEncoder
from losses import MSELoss
from callbacks import TestDecoder
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_fn(dataset, input_size=(28, 28)):
    x = tf.cast(dataset['image'], tf.float32)
    x = tf.image.resize(x, input_size)
    x = x / 255.
    return x, x


dataset = 'mnist'     # 'cifar10', 'fashion_mnist', 'mnist'
log_dirs = 'logs_vae'
batch_size = 64
latent_dim = 2

# Load datasets and setting
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
# train_data = tfds.load(dataset, split=combine_split, data_dir='/home/share/dataset/tensorflow-datasets')
train_data = tfds.load(dataset, split=tfds.Split.TRAIN)
train_data = train_data.shuffle(1000)
train_data = train_data.map(parse_fn, num_parallel_calls=AUTOTUNE)
train_data = train_data.batch(batch_size)
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
test_data = tfds.load(dataset, split=tfds.Split.TEST)
test_data = test_data.map(parse_fn, num_parallel_calls=AUTOTUNE)
test_data = test_data.batch(batch_size)
test_data = test_data.prefetch(buffer_size=AUTOTUNE)

# Callbacks function
model_dir = log_dirs + '/models'
os.makedirs(model_dir, exist_ok=True)
model_tb = tf.keras.callbacks.TensorBoard(log_dir=log_dirs)
model_mckp = tf.keras.callbacks.ModelCheckpoint(model_dir + '/best_model.h5',
                                                monitor='val_loss',
                                                save_best_only=True,
                                                mode='min')
model_testd = TestDecoder(28, log_dir=log_dirs)

# Create model
vae = VariationalAutoEncoder(latent_dim=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=MSELoss())
vae.fit(train_data, epochs=50, validation_data=test_data, callbacks=[model_tb, model_mckp, model_testd])
