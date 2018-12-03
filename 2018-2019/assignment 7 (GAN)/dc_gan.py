import numpy as np
import tensorflow as tf
from gan import GAN

class DCGAN(GAN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, gan_type='DCGAN')

    def generator(self, z):
        img_h, img_w, img_c = self.img_hwc

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            # l1
            fc1_units = 1024
            fc1 = tf.layers.dense(z, fc1_units, activation=tf.nn.relu)
            bn1 = tf.layers.batch_normalization(fc1, training=True)

            # l2
            fc2_units = (img_h // 4) * (img_w // 4) * 128
            fc2 = tf.layers.dense(bn1, fc2_units, activation=tf.nn.relu)
            bn2 = tf.layers.batch_normalization(fc2, training=True)

            new_shape = [-1, img_h // 4, img_w // 4, 128]
            bn2_reshaped = tf.reshape(bn2, new_shape)

            # l3
            conv3 = tf.layers.conv2d_transpose(bn2_reshaped, 
                                               filters=64, kernel_size=4,
                                               strides=2, padding='same',
                                               activation=tf.nn.relu)
            bn3 = tf.layers.batch_normalization(conv3, training=True)

            # l4
            conv4 = tf.layers.conv2d_transpose(bn3,
                                               filters=img_c, kernel_size=4,
                                               strides=2, padding='same')
            img = tf.nn.tanh(conv4)
            return img

    def discriminator(self, x):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            # l1
            x_pad = tf.pad(x, [[0,0], [0,1], [0,1], [0,0]], 'constant')
            conv1 = tf.layers.conv2d(x_pad, filters=32, kernel_size=5, strides=2)
            a1 = leaky_relu(conv1, 0.01)

            # l2
            conv2 = tf.layers.conv2d(a1, filters=64, kernel_size=5, strides=2)
            a2 = leaky_relu(conv2, 0.01)
            a2_flat = tf.layers.flatten(a2)

            # l3
            # + 1 for aux padding
            h = (((self.img_hwc[0] + 1 - 5) // 2 + 1) - 5) // 2 + 1
            w = (((self.img_hwc[1] + 1 - 5) // 2 + 1) - 5) // 2 + 1
            fc3 = tf.layers.dense(a2_flat, h * w * 64)
            a3 = leaky_relu(fc3, 0.01)

            # l4
            logits = tf.layers.dense(a3, 1)
            return logits

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)