import numpy as np
import tensorflow as tf
from gan import GAN

class VanillaGAN(GAN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, gan_type='VanillaGAN')

    def generator(self, z):
        img_h, img_w, img_c = self.img_hwc

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            fc1 = tf.layers.dense(z, 1024, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
            fc3 = tf.layers.dense(fc2, self.img_dim, activation=tf.nn.tanh)
            img = tf.reshape(fc3, [-1, img_h, img_w, img_c])
            return img

    def discriminator(self, x):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            x_reshaped = tf.reshape(x, [-1, self.img_dim])
            fc1 = tf.layers.dense(x_reshaped, 256)
            a1 = leaky_relu(fc1, 0.01)

            fc2 = tf.layers.dense(a1, 256)
            a2 = leaky_relu(fc2, 0.01)

            logits = tf.layers.dense(a2, 1)
            return logits

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)