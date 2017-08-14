import tensorflow as tf
import RBM

class DBN:

    def __init__(self, weights, biases):
        self.ws = weights
        self.bs = biases

    def gen_sample(self, k, x=None):
        # Get input value for the top rbm's visible layer
        if x is None:
            x = tf.zeros([1, int(self.bs[-2].shape[-1])])
        else:
            # Propagate input value up to visible layer of top rbm
            for i in range(len(self.ws) - 1):
                x_prob = tf.sigmoid(tf.matmul(x, self.ws[i]) + self.bs[i+1])
                x = RBM.sample(x_prob)

        # Sample from the top layer
        top_rbm = RBM.RBM(self.ws[-1], self.bs[-2], self.bs[-1])
        x_sample = top_rbm.gibbs_sample(x, k)

        # Propagate sample down to visible layer
        for i in reversed(range(len(self.ws) - 1)):
            x_prob = tf.sigmoid(tf.matmul(x_sample, tf.transpose(self.ws[i])) + self.bs[i])
            x_sample = RBM.sample(x_prob)

        return x_sample
