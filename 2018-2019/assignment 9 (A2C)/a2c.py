import config
import tensorflow as tf

class A2C:
    def __init__(self, action_size):
        self.state_shape = config.STATE_SHAPE
        self.action_size = action_size
        
        with tf.variable_scope('A2C'):
            self.input_ph = tf.placeholder(tf.float32,
                                           [None, *self.state_shape], name='input_ph')
            self.action_ph = tf.placeholder(tf.uint8, 
                                            (), name='action_ph')
            self.target_ph = tf.placeholder(tf.float32,
                                            (), name='td_target_ph')
            
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
            
            # actor
            with tf.name_scope('actor'):
                self.predicted_logits = tf.layers.dense(self.fc4,
                                                        units=self.action_size,
                                                        kernel_initializer=initializer,
                                                        activation=None,
                                                        name='predicted_logits')
            
                self.predicted_probs = tf.nn.softmax(self.predicted_logits)
            
            # critic
            with tf.name_scope('critic'):
                self.predicted_value = tf.layers.dense(self.fc4,
                                                       units=1,
                                                       kernel_initializer=initializer,
                                                       activation=None,
                                                       name='predicted_value')
            
            # td-error
            with tf.name_scope('td_error'):
                self.td_error = self.predicted_value - tf.stop_gradient(self.target_ph)
            
            # loss
            with tf.name_scope('loss'):
                x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits
                self.actor_loss = x_entropy(labels=[tf.to_int32(self.action_ph)],
                                            logits=self.predicted_logits)\
                                  * tf.stop_gradient(self.td_error)
                        
                self.critic_loss = self.td_error ** 2
            
            # optimization
            with tf.name_scope('optimization'):
                self.optimizer = tf.train.RMSPropOptimizer(config.LEARNING_RATE)
                self.actor_optimization_step = self.optimizer.minimize(self.actor_loss)
                self.critic_optimization_step = self.optimizer.minimize(self.critic_loss)