import numpy as np
import tensorflow as tf
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


size = 28
n = 15
save_images = np.zeros((size * n, size * n, 1))
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)
model = tf.keras.models.load_model('logs_vae/models/best_model.h5')
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        img = model(z_sample)
        save_images[i * size: (i + 1) * size, j * size: (j + 1) * size] = img.numpy()[0]

plt.imshow(save_images[..., 0], cmap='gray')
plt.show()
