import numpy as np 
import tensorflow as tf

# defines a convolutional layer
# from: https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py
def conv(x, scope, *, nf, rf, stride, activ_fn=tf.nn.relu, pad='VALID', init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[3].value
        w = tf.get_variable("w", [rf, rf, nin, nf], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nf], initializer=tf.constant_initializer(0.0))
        activations = activ_fn(tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad)+b)
        tf.summary.histogram("Weights", w)
        tf.summary.histogram("Biases", b)
        tf.summary.histogram("Activations", activations)
        return activations

# defines a fully connected layer
# from: https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py
def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        activations = tf.matmul(x, w)+b
        tf.summary.histogram("Weights", w)
        tf.summary.histogram("Biases", b)
        tf.summary.histogram("Activations", activations)
        return activations

# Takes output from conv and converts it to fc (flattens) 
# from: https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py
def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

class Model():
    def __init__(self, obs_shape, act_size, LR, cuda, log_str):
        # Number of possible actions
        self.act_size = act_size
        # Shape of the observations
        self.obs_shape = obs_shape
        # Dummy observation of the correct shape 
        # to do shape manipulations
        dummy_obs = np.ndarray(obs_shape)
        flat_obs = dummy_obs.reshape(-1)

        # Neural Network that defines the policy
        with tf.name_scope("Inputs"):
            self.obs = tf.placeholder(tf.float32, shape=(None,)+ obs_shape)

        with tf.name_scope("FeatureExtractor"):
            h = conv(self.obs, 'FE1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'FE2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'FE3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))

        with tf.name_scope("FeatureUser"):
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'FC1', nh=512, init_scale=np.sqrt(2))
            self.pi = fc(h4, 'pi', act_size, init_scale=0.01)
            self.vf = fc(h4, 'v', 1)[:,0]

        with tf.name_scope("Stochastic"):
            distrib = tf.contrib.distributions.Categorical(probs=self.pi)
            # Samples from the categorical distribution to determine action to take
            self.act_taken = distrib.sample()


        if(cuda):
            SESS_CONFIG = tf.ConfigProto(device_count = {'GPU': 1})
        else:
            SESS_CONFIG = tf.ConfigProto(device_count = {'GPU': 0})

        # Create the session
        self.sess = tf.Session(config=SESS_CONFIG)

        # Init all weights
        self.sess.run(tf.global_variables_initializer())

        # Merge Summaries and Create Summary Writer for TB
        all_summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(log_str)
        self.writer.add_graph(self.sess.graph) 




    def forward(self, x):
        """Takes in an observation and returns action probabilities and
        an estimate of the maximum discounted reward attainable 
        from the current state"""

        act_probs, values = self.sess.run([self.pi, self.vf], feed_dict={self.obs: x})
        return act_probs, values


    def act_stochastic(self, obs):
        """Returns an action chosen semi stochastically"""

        act_taken, act_probs = self.sess.run([self.act_taken, self.pi], feed_dict={self.obs: obs})
        act_taken_v = np.zeros(self.act_size)
        act_taken_v[act_taken[0]] = 1
        return act_taken, act_taken_v, act_probs

    def learn(self, replay_buffer):
        """Performs backprop w.r.t. the replay buffer"""
        return 0


        




