import config
import tensorflow as tf

class DQN:
    def __init__(self, action_size, dqn_name='DQN'):
        self.state_shape = config.STATE_SHAPE
        self.action_size = action_size
        
        with tf.variable_scope(dqn_name):
            self.input_ph = tf.placeholder(tf.float32,
                                           [None, *self.state_shape], name='input')
            self.actions_ph = tf.placeholder(tf.uint8, 
                                             [None], name='action')
            self.Q_target_ph = tf.placeholder(tf.float32,
                                             [None], name='target')
            
            self.actions_one_hot = tf.one_hot(self.actions_ph, self.action_size)
            initializer = tf.initializers.he_uniform()
            
            # l1
            self.conv1 = tf.layers.conv2d(self.input_ph,
                                          filters=32, kernel_size=8,
                                          strides=4, padding='valid',
                                          kernel_initializer=initializer,
                                          activation=tf.nn.relu, name='conv1')
            
            # l2
            self.conv2 = tf.layers.conv2d(self.conv1,
                                          filters=64, kernel_size=4,
                                          strides=2, padding='valid',
                                          kernel_initializer=initializer,
                                          activation=tf.nn.relu, name='conv2')
            
            # l3
            self.conv3 = tf.layers.conv2d(self.conv2,
                                          filters=64, kernel_size=3,
                                          strides=1, padding='valid',
                                          kernel_initializer=initializer,
                                          activation=tf.nn.relu, name='conv3')
            self.flatten3 = tf.layers.flatten(self.conv3)
            
            # l4
            self.fc4 = tf.layers.dense(self.flatten3,
                                       units=512,
                                       kernel_initializer=initializer,
                                       activation=tf.nn.relu, name='fc4')
            
            # l5
            self.output = tf.layers.dense(self.fc4,
                                          units=self.action_size,
                                          kernel_initializer=initializer,
                                          activation=None, name='output')
            
            # prediction
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_one_hot), axis=1)
            
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(tf.square(self.Q_target_ph - self.Q))
            
            with tf.name_scope('optimization'):
                self.optimizer = tf.train.RMSPropOptimizer(config.LEARNING_RATE,
                                                           decay=config.DECAY,
                                                           momentum=config.MOMENTUM,
                                                           epsilon=config.EPSILON,
                                                           centered=config.CENTERED)
                self.optimization_step = self.optimizer.minimize(self.loss)