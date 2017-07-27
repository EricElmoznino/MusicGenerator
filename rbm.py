import tensorflow as tf
from tensorflow.python.ops import control_flow_ops as cf

class rbm:
    def __init__(self, weights, bias_visible, bias_hidden):
        self.w = weights
        self.bv = bias_visible
        self.bh = bias_hidden

    def __sample(self, probabilities):
        return tf.floor(probabilities + tf.random_uniform(tf.shape(probabilities), 0, 1))

    def gibbs_sample(self, inputs, num_iterations, trainable=True, swap_mem=False):
        # Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
        def gibbs_step(i, k, xk):
            # Runs a single gibbs step. The visible values are initialized to xk
            hk = self.__sample(tf.sigmoid(tf.matmul(xk, self.w) + self.bh))
            xk = self.__sample(tf.sigmoid(tf.matmul(hk, tf.transpose(self.w)) + self.bv))
            return i+1 + 1, k, xk

        count = tf.constant(0)
        _, _, x_sample = cf.while_loop(lambda i, n, *args: i < n,
                                       gibbs_step, [count, num_iterations, inputs],
                                       back_prop=trainable, swap_memory=swap_mem)
        return x_sample

