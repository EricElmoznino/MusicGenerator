import tensorflow as tf
from RBM import RBM
import Helpers as hp

class RNN_DBN:

    def __init__(self, visible_size, hidden_sizes, state_size, num_rnn_cells=1):
        self.dbn_sizes = [visible_size] + hidden_sizes
        self.s_size = state_size
        self.num_rnn_cells = num_rnn_cells

        with tf.variable_scope('dbn'):
            self.W = []
            for i in range(len(self.dbn_sizes) - 1):
                v = self.dbn_sizes[i]
                h = self.dbn_sizes[i+1]
                self.W.append(hp.weight_variables([v, h], stddev=0.01, name='layer_'+str(i+1)))

        with tf.variable_scope('rnn'):
            if num_rnn_cells > 1:
                self.rnn = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.BasicLSTMCell(self.s_size) for _ in range(num_rnn_cells)]
                )
            else:
                self.rnn = tf.contrib.rnn.BasicLSTMCell(self.s_size)
            self.rnn_s0 = self.rnn.zero_state(1, tf.float32)

        with tf.variable_scope('rnn_to_dbn'):
            self.Wu = []
            self.Wq = []
            self.B = []
            for i, s in enumerate(self.dbn_sizes):
                self.Wu.append(hp.weight_variables([self.s_size, s], stddev=0.0001, name='weights_u'+str(i+1)))
                self.Wq.append(hp.weight_variables([self.s_size, s], stddev=0.0001, name='weights_q'+str(i+1)))
                self.B.append(hp.bias_variables([self.s], value=0.0, name='biases_'+str(i+1)))

    def generation_model(self, x, length):
        with tf.variable_scope('generation'):
            primer_states = self.__unroll_rnn(x)
            if self.num_rnn_cells > 1:
                primer_state = tuple([tf.contrib.rnn.LSTMStateTuple(state.c[-1], state.h[-1]) for state in primer_states])
            else:
                primer_state = tf.contrib.rnn.LSTMStateTuple(primer_states.c[-1], primer_states.h[-1])

        def music_timestep(t, k, x_t, s_tm1, music):
            if self.num_rnn_cells > 1:
                u_tm1 = s_tm1[-1].c
                q_tm1 = s_tm1[-1].h
            else:
                u_tm1 = s_tm1.c
                q_tm1 = s_tm1.h
            bh = tf.matmul(u_tm1, self.Wu[1]) + tf.matmul(q_tm1, self.Wq[1]) + self.B[1]
            bv = tf.matmul(u_tm1, self.Wu[0]) + tf.matmul(q_tm1, self.Wq[0]) + self.B[0]
            rbm = RBM(self.W[0], bv, bh)
            _, notes_t = rbm.gibbs_sample(x_t, 25)
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
            states = self.__unroll_rnn(x)
            state0 = self.rnn_s0
            if self.num_rnn_cells > 1:
                states = states[-1]
                state0 = state0[-1]
            u_t = tf.reshape(states.c, [-1, self.s_size])
            q_t = tf.reshape(states.h, [-1, self.s_size])
            u_tm1 = tf.concat([state0.c, u_t], 0)[:-1, :]
            q_tm1 = tf.concat([state0.h, q_t], 0)[:-1, :]
            bh = tf.matmul(u_tm1, self.Wu[1]) + tf.matmul(q_tm1, self.Wq[1]) + self.B[1]
            bv = tf.matmul(u_tm1, self.Wu[0]) + tf.matmul(q_tm1, self.Wq[0]) + self.B[0]
            rbm = RBM(self.W[0], bv, bh)

        with tf.variable_scope('train_ops'):
            cost, loglikelihood = rbm.free_energy_cost(x, 15)
            cost_summary = tf.summary.scalar('free_energy', cost)
            ll_summary = tf.summary.scalar('log_likelihood', loglikelihood)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            gradients = optimizer.compute_gradients(cost)
            gradients = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gradients]
            optimizer = optimizer.apply_gradients(gradients)

        return cost, optimizer, loglikelihood, [cost_summary, ll_summary]

    def pretrain_model(self, x):
        with tf.variable_scope('pre-train_rbm'):
            rbm = RBM(self.W[0], self.B[0], self.B[1])
        with tf.variable_scope('pre-train_ops'):
            cost, _ = rbm.free_energy_cost(x, 1)
            optimizer = tf.train.AdamOptimizer().minimize(cost)
        return cost, optimizer

    def __unroll_rnn(self, x):
        def recurrence(s_tm1, _x):
            _x = tf.reshape(_x, [1, self.dbn_sizes[0]])
            _, s_t = self.rnn(_x, s_tm1)
            return s_t
        states = tf.scan(recurrence, x, initializer=self.rnn_s0)
        return states
