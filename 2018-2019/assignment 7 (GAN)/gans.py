import tensorflow as tf

class DCGAN:
    def __init__(self, z_dim=50, img_h=28, img_w=28, img_c=1):
        self.z_dim = z_dim
        self.img_hwc = (img_h, img_w, img_c)

    def generator(self, z):
        '''
          Inputs:
          - z: Tensor of random noise with shape [batch_size, z_dim] 
          
          Returns:
          - img: Tensor of generated images with shape [batch_size, 
                                                        img_h, img_w, img_c]
        '''
        img_h, img_w, img_c = self.img_hwc

        with tf.variable_scope('generator'):
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
        '''
          Inputs:
          - x: Tensor of input images with shape [batch_size, 
                                                  img_h, img_w, img_c]
          
          Returns:
          - logits: Tensor containing the score for an image being real
                    for each input image with shape [batch_size, 1]
        '''
        with tf.variable_scope('discriminator'):
            # l1
            conv1 = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=2)
            a1 = leaky_relu(conv1, 0.01)

            # l2
            conv2 = tf.layers.conv2d(a1, filters=64, kernel_size=5, strides=2)
            a2 = leaky_relu(conv2, 0.01)
            a2_flat = tf.layers.flatten(a2)

            # l3
            h = (((self.img_hwc[0] - 5) // 2 + 1) - 5) // 2 + 1
            w = (((self.img_hwc[1] - 5) // 2 + 1) - 5) // 2 + 1
            fc3 = tf.layers.dense(a2_flat, h * w * 64)
            a3 = leaky_relu(fc3, 0.01)

            # l4
            logits = tf.layers.dense(a3, 1)
            return logits

    def generate_images(self, batch_size):
        z = sample_noise(batch_size, self.z_dim)
        return self.generator(z)

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha * x)

def sample_noise(batch_size, dim):
    return tf.random_uniform((batch_size, dim), -1, 1)