import numpy as np
import tensorflow as tf
from vis import grid_vis

class GAN:
    def __init__(self, z_dim=50, img_h=28, img_w=28, img_c=1, dataset_name='', gan_type=''):
        self.z_dim = z_dim
        self.img_hwc = (img_h, img_w, img_c)
        self.img_dim = img_h * img_w * img_c
        self.dataset_name = dataset_name
        self.gan_type = gan_type

    def generator(self, z):
        '''
          Inputs:
          - z: Tensor of random noise with shape [batch_size, z_dim] 
          
          Returns:
          - img: Tensor of generated images with shape [batch_size, 
                                                        img_h, img_w, img_c]
        '''
        pass

    def discriminator(self, x):
        '''
          Inputs:
          - x: Tensor of input images with shape [batch_size, 
                                                  img_h, img_w, img_c]
          
          Returns:
          - logits: Tensor containing the score for an image being real
                    for each input image with shape [batch_size, 1]
        '''
        pass

    def discriminator_loss(self, logits_real, logits_fake):
        '''
          Compute the discriminator loss.

          Inputs:
          - logits_real: Unnormalized score that the image is real
                         for each real image. Tensor of shape [batch_size, 1].
          - logits_fake: Unnormalized score that the image is real 
                         for each fake image. Tensor of shape [batch_size, 1].
          
          Returns:
          - D_loss: discriminator loss scalar
        '''
        F_xent = tf.nn.sigmoid_cross_entropy_with_logits
        D_loss_r = tf.reduce_mean(F_xent(labels=tf.ones_like(logits_real),
                                         logits=logits_real))
        D_loss_f = tf.reduce_mean(F_xent(labels=tf.zeros_like(logits_fake),
                                         logits=logits_fake))
        D_loss = D_loss_r + D_loss_f
        return D_loss

    def generator_loss(self, logits_fake):
        '''
          Compute the generator loss.

          Inputs:
          - logits_fake: Unnormalized score that the image is real 
                         for each fake image. Tensor of shape [batch_size, 1].
          
          Returns:
          - G_loss: generator loss scalar
        '''
        F_xent = tf.nn.sigmoid_cross_entropy_with_logits
        G_loss = tf.reduce_mean(F_xent(labels=tf.ones_like(logits_fake),
                                       logits=logits_fake))
        return G_loss

    def generate_images(self, batch_size):
        z = sample_noise(batch_size, self.z_dim)
        return self.generator(z)

    def generate_images_np(self, num_images):
        # do not use this function during training
        tf.reset_default_graph()
        generate_images = self.generate_images(num_images)
        with tf.Session() as sess:
            model_name = f'./saved_models/{self.gan_type}_{self.dataset_name}.ckpt'
            tf.train.Saver().restore(sess, model_name)
            generated_images = sess.run(generate_images)
        return generated_images

    def train(self, X, batch_size=128, num_epochs=5, print_every=None, lr=1e-3, beta1=0.5):
        tf.reset_default_graph()

        with tf.name_scope('input'):
            h, w, c = self.img_hwc
            x_real = tf.placeholder(tf.float32, shape=[None, h, w, c])
            batch_size_placeholder = tf.placeholder(tf.int32)

        with tf.name_scope('loss'):
            logits_real = self.discriminator(x_real)
            x_fake = self.generate_images(batch_size_placeholder)
            logits_fake = self.discriminator(x_fake)
            D_loss = self.discriminator_loss(logits_real, logits_fake)
            G_loss = self.generator_loss(logits_fake)

        with tf.name_scope('optimizers'):
            D_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
            G_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
        
        with tf.name_scope('training_steps'):
            D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
            G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator') 
            D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
            G_train_step = G_solver.minimize(G_loss, var_list=G_vars)

        N = X.shape[0]
        start_indices = list(range(0, N, batch_size))
        end_indices = list(range(batch_size, N, batch_size))
        if (len(end_indices) == 0) or (end_indices[-1] != N):
            end_indices.append(N)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print(f'Epoch 0')
            generated_images = sess.run(x_fake, feed_dict={batch_size_placeholder: 45})
            grid_vis(generated_images, img_h=h, img_w=w, img_c=c, rows=3, cols=15)

            D_loss_per_epoch = [0]
            G_loss_per_epoch = [0]

            for epoch in range(num_epochs):
                D_loss_per_batch = []
                G_loss_per_batch = []

                for start_idx, end_idx in zip(start_indices, end_indices):
                    X_batch = X[start_idx:end_idx]
                    
                    _, D_loss_batch = sess.run([D_train_step, D_loss],
                                               feed_dict={
                                                    x_real: X_batch,
                                                    batch_size_placeholder: len(X_batch)
                                                })
                    _, G_loss_batch = sess.run([G_train_step, G_loss],
                                               feed_dict={batch_size_placeholder: batch_size})

                    D_loss_per_batch.append(D_loss_batch)
                    G_loss_per_batch.append(G_loss_batch)

                D_loss_per_epoch.append(np.mean(D_loss_per_batch))
                G_loss_per_epoch.append(np.mean(G_loss_per_batch))

                if (print_every is not None) and ((epoch + 1) % print_every == 0):
                    print(f'Epoch: {epoch + 1}, '
                          f'D_loss: {D_loss_per_epoch[epoch + 1]:6.4f}, '
                          f'G_loss: {G_loss_per_epoch[epoch + 1]:6.4f}')

                    generated_images = sess.run(x_fake, feed_dict={batch_size_placeholder: 45})
                    grid_vis(generated_images, img_h=h, img_w=w, img_c=c, rows=3, cols=15)

            tf.train.Saver().save(sess, f'./saved_models/{self.gan_type}_{self.dataset_name}.ckpt')
        return D_loss_per_epoch, G_loss_per_epoch

def sample_noise(batch_size, dim):
    return tf.random_uniform((batch_size, dim), -1, 1)