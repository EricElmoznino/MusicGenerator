import tensorflow as tf
from tensorflow.python.ops import control_flow_ops as cf


def sample(probabilities):
    return tf.floor(probabilities + tf.random_uniform(tf.shape(probabilities), 0, 1))


class RBM:

    def __init__(self, weights, bias_visible, bias_hidden):
        self.w = weights
        self.bv = bias_visible
        self.bh = bias_hidden

    def gibbs_step(self, i, k, x_prob, xk):
        # Runs a single gibbs step. The visible values are initialized to xk
        h_prob = tf.sigmoid(tf.matmul(xk, self.w) + self.bh)
        hk = sample(h_prob)
        x_prob = tf.sigmoid(tf.matmul(hk, tf.transpose(self.w)) + self.bv)
        xk = sample(x_prob)
        return i + 1, k, x_prob, xk

    def gibbs_sample(self, x, num_iterations):
        count = tf.constant(0)
        _, _, x_prob, x_sample = cf.while_loop(lambda i, n, *args: i < n,
                                               self.gibbs_step, [count, num_iterations, x, x],
                                               back_prop=False)
        return x_prob, x_sample

    def free_energy_cost(self, x, num_iterations):
        _, x_sample = self.gibbs_sample(x, num_iterations)

        def free_energy(_x):
            return -tf.reduce_sum(tf.log1p(tf.exp(tf.matmul(_x, self.w) + self.bh)), 1) - \
                   tf.reduce_sum(tf.multiply(_x, self.bv), 1)
        cost = tf.reduce_mean(free_energy(x) - free_energy(x_sample))

        _, _, x_prob, _ = self.gibbs_step(0, 0, x_sample, x_sample)
        safe_xprob = tf.where(tf.equal(x_prob, 0.0), tf.ones_like(x_prob), x_prob)
        safe_1mxprob = tf.where(tf.equal(1.0-x_prob, 0.0), tf.ones_like(1.0-x_prob), 1.0-x_prob)
        loglikelihood = tf.multiply(x, tf.log(safe_xprob)) + tf.multiply(1.0-x, tf.log(safe_1mxprob))
        loglikelihood = tf.reduce_mean(tf.reduce_sum(loglikelihood, 1))

        return cost, loglikelihood

