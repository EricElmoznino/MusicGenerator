import tensorflow as tf
from RBM import RBM
import Helpers as hp

class RNN_RBM:

    def __init__(self, visible_size, hidden_size, state_size, use_lstm=False, num_rnn_cells=1):
        self.v_size = visible_size
        self.h_size = hidden_size
        self.s_size = state_size
        self.use_lstm = use_lstm
        self.num_rnn_cells = num_rnn_cells

        with tf.variable_scope('rbm'):
            self.W = hp.weight_variables([self.v_size, self.h_size], stddev=0.01)

        with tf.variable_scope('rnn'):
            def cell():
                if use_lstm:
                    return tf.contrib.rnn.BasicLSTMCell(self.s_size)
                else:
                    return tf.contrib.rnn.BasicRNNCell(self.s_size)
            if num_rnn_cells > 1:
                self.rnn = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(num_rnn_cells)])
            else:
                self.rnn = cell()
            self.rnn_s0 = self.rnn.zero_state(1, tf.float32)

        with tf.variable_scope('rnn_to_rbm'):
            self.Wuh = hp.weight_variables([self.s_size, self.h_size], stddev=0.0001, name='weights_h')
            self.Wuv = hp.weight_variables([self.s_size, self.v_size], stddev=0.0001, name='weights_v')
            self.Buh = hp.bias_variables([self.h_size], value=0.0, name='biases_h')
            self.Buv = hp.bias_variables([self.v_size], value=0.0, name='biases_v')

    def generation_model(self, x, length):
        with tf.variable_scope('generation'):
            _, primer_state = self.__unroll_rnn(x)

        def music_timestep(t, k, x_t, s_tm1, music):
            _s_tm1 = s_tm1[-1] if self.num_rnn_cells > 1 else s_tm1
            _s_tm1 = _s_tm1.h if self.use_lstm else _s_tm1
            bh = tf.matmul(_s_tm1, self.Wuh) + self.Buh
            bv = tf.matmul(_s_tm1, self.Wuv) + self.Buv
            rbm = RBM(self.W, bv, bh)
            notes_t = rbm.gibbs_sample(x_t, 25)
            _, s_t = self.rnn(notes_t, s_tm1)
            music = music + tf.concat([tf.zeros([t, self.v_size]), notes_t,
                                       tf.zeros([k-t-1, self.v_size])], 0)
            return t+1, k, notes_t, s_t, music

        count = tf.constant(0)
        music = tf.zeros([length, self.v_size])
        _, _, _, _, music = tf.while_loop(lambda t, k, *args: t < k, music_timestep,
                                          [count, length, tf.zeros([1, self.v_size]), primer_state, music],
                                          back_prop=False)
        return music

    def train_model(self, x):
        with tf.variable_scope('train_rnn_rbm'):
            states, _ = self.__unroll_rnn(x)
            state0 = self.rnn_s0[-1] if self.num_rnn_cells > 1 else self.rnn_s0
            state0 = state0.h if self.use_lstm else state0
            states_tm1 = tf.concat([state0, states], 0)[:-1, :]
            bh = tf.matmul(states_tm1, self.Wuh) + self.Buh
            bv = tf.matmul(states_tm1, self.Wuv) + self.Buv
            rbm = RBM(self.W, bv, bh)

        with tf.variable_scope('train_ops'):
            cost = rbm.free_energy_cost(x, 15)
            cost_summary = tf.summary.scalar('train_cost', cost)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            gradients = optimizer.compute_gradients(cost)
            gradients = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gradients]
            optimizer = optimizer.apply_gradients(gradients)

        return cost, optimizer, cost_summary

    def pretrain_model(self, x):
        with tf.variable_scope('pre-train_rbm'):
            rbm = RBM(self.W, self.Buv, self.Buh)
        with tf.variable_scope('pre-train_ops'):
            cost = rbm.free_energy_cost(x, 1)
            optimizer = tf.train.AdamOptimizer().minimize(cost)
        return cost, optimizer

    def __unroll_rnn(self, x):
        x = tf.reshape(x, [1, -1, self.v_size])
        states, final_state = tf.nn.dynamic_rnn(self.rnn, x, initial_state=self.rnn_s0)
        states = tf.reshape(states, [-1, self.s_size])
        return states, final_state

