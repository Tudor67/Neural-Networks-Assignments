import tensorflow as tf


class DQN:
    def __init__(self, state_shape, action_size, dqn_name='DQN'):
        self.state_shape = state_shape
        self.action_size = action_size
        
        with tf.variable_scope(dqn_name):
            self.input_ph = tf.placeholder(tf.float32,
                                           [None, *self.state_shape], name='input')
            self.action_indices_ph = tf.placeholder(tf.uint8,
                                                    [None], name='action')
            self.Q_target_ph = tf.placeholder(tf.float32,
                                             [None], name='target')
            self.learning_rate_ph = tf.placeholder(tf.float32,
                                                   (), name='learning_rate')
            
            self.actions_one_hot = tf.one_hot(self.action_indices_ph, self.action_size)
            xavier_init = tf.glorot_uniform_initializer()
            
            # l1
            self.conv1 = tf.layers.conv2d(self.input_ph,
                                          filters=32, kernel_size=8,
                                          strides=4, padding='valid',
                                          kernel_initializer=xavier_init,
                                          activation=tf.nn.elu, name='conv1')
            
            # l2
            self.conv2 = tf.layers.conv2d(self.conv1,
                                          filters=64, kernel_size=4,
                                          strides=2, padding='valid',
                                          kernel_initializer=xavier_init,
                                          activation=tf.nn.elu, name='conv2')
            
            # l3
            self.conv3 = tf.layers.conv2d(self.conv2,
                                          filters=64, kernel_size=3,
                                          strides=2, padding='valid',
                                          kernel_initializer=xavier_init,
                                          activation=tf.nn.elu, name='conv3')
            self.flatten3 = tf.layers.flatten(self.conv3)
            
            # l4
            self.fc4 = tf.layers.dense(self.flatten3,
                                       units=512,
                                       kernel_initializer=xavier_init,
                                       activation=tf.nn.elu, name='fc4')
            
            # l5
            self.output = tf.layers.dense(self.fc4,
                                          units=self.action_size,
                                          kernel_initializer=xavier_init,
                                          activation=None, name='output')
            
            # prediction
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_one_hot), axis=1)
            
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.square(self.Q_target_ph - self.Q))
            
            with tf.name_scope('optimization'):
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate_ph)
                self.optimization_step = self.optimizer.minimize(self.loss)