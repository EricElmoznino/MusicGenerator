import tensorflow as tf
from tensorflow.python.ops import control_flow_ops as cf


class RBM:

    def __init__(self, weights, bias_visible, bias_hidden):
        self.w = weights
        self.bv = bias_visible
        self.bh = bias_hidden

    def __sample(self, probabilities):
        return tf.floor(probabilities + tf.random_uniform(tf.shape(probabilities), 0, 1))

    def gibbs_sample(self, x, num_iterations, trainable=True, swap_mem=False):
        # Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
        def gibbs_step(i, k, xk):
            # Runs a single gibbs step. The visible values are initialized to xk
            hk = self.__sample(tf.sigmoid(tf.matmul(xk, self.w) + self.bh))
            xk = self.__sample(tf.sigmoid(tf.matmul(hk, tf.transpose(self.w)) + self.bv))
            return i+1, k, xk

        count = tf.constant(0)
        _, _, x_sample = cf.while_loop(lambda i, n, *args: i < n,
                                       gibbs_step, [count, num_iterations, x],
                                       back_prop=trainable, swap_memory=swap_mem)
        return x_sample

    def free_energy_cost(self, x, num_iterations):
        x_sample = self.gibbs_sample(x, num_iterations)

        def free_energy(_x):
            return -tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(_x, self.w) + self.bh)), 1) - \
                   tf.reduce_sum(tf.multiply(_x, self.bv), 1)

        return tf.reduce_mean(free_energy(x) - free_energy(x_sample))

    def apply_cd(self, x, lr):
        x_sample = self.gibbs_sample(x, 1)
        h = self.__sample(tf.sigmoid(tf.matmul(x, self.w) + self.bh))
        h_sample = self.__sample(tf.sigmoid(tf.matmul(x_sample, self.w) + self.bh))

        lr = tf.constant(lr, dtype=tf.float32)
        batch_size = tf.cast(tf.shape(x)[0], tf.float32)
        w_delta = tf.multiply(lr/batch_size, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
        bv_delta = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
        bh_delta = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))

        optimizer = [self.w.assign_add(w_delta),
                     self.bv.assign_add(tf.reshape(bv_delta, [-1])),
                     self.bh.assign_add(tf.reshape(bh_delta, [-1]))]
        return optimizer

