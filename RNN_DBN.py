import tensorflow as tf
from RBM import RBM, sample
from DBN import DBN
import Helpers as hp


# Provides training and generation subgraphs for an rnn deep belief net
class RNN_DBN:

    def __init__(self, visible_size, hidden_sizes, state_size, num_rnn_cells=1):
        self.v_size = visible_size
        self.dbn_sizes = [visible_size] + hidden_sizes  # the layer sizes of the dbn
        self.s_size = state_size
        self.num_rnn_cells = num_rnn_cells

        # Model variables

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
                self.B.append(hp.bias_variables([s], value=0.0, name='biases_'+str(i+1)))

    def generation_model(self, x, length):
        with tf.variable_scope('generation'):
            # Propagate the states of a primer song and remember the
            # final state. It is used as the initial state of the generated song
            with tf.variable_scope('primer'):
                primer_states = self.__unroll_rnn(x)
                if self.num_rnn_cells > 1:
                    primer_state = tuple([tf.contrib.rnn.LSTMStateTuple(state.c[-1], state.h[-1]) for state in primer_states])
                else:
                    primer_state = tf.contrib.rnn.LSTMStateTuple(primer_states.c[-1], primer_states.h[-1])

            def music_timestep(t, k, x_tm1, s_tm1, music):
                with tf.variable_scope('generation_timestep'):
                    if self.num_rnn_cells > 1:
                        u_tm1 = s_tm1[-1].c
                        q_tm1 = s_tm1[-1].h
                    else:
                        u_tm1 = s_tm1.c
                        q_tm1 = s_tm1.h

                    # Use the previous lstm state to compute the biases
                    # for each layer of the dbn
                    dbn_biases = []
                    for wu, wq, b in zip(self.Wu, self.Wq, self.B):
                        dbn_biases.append(tf.matmul(u_tm1, wu) + tf.matmul(q_tm1, wq) + b)

                    dbn = DBN(self.W, dbn_biases)
                    notes_t = dbn.gen_sample(25, x=x_tm1)

                    _, s_t = self.rnn(notes_t, s_tm1)

                    # Concatenate the current music timestep to the whole song
                    music = music + tf.concat([tf.zeros([t, self.v_size]), notes_t,
                                               tf.zeros([k-t-1, self.v_size])], 0)
                return t+1, k, notes_t, s_t, music

            with tf.variable_scope('generation_loop'):
                count = tf.constant(0)
                music = tf.zeros([length, self.v_size])
                _, _, _, _, music = tf.while_loop(lambda t, k, *args: t < k, music_timestep,
                                                  [count, length, tf.zeros([1, self.v_size]), primer_state, music],
                                                  back_prop=False)
        return music

    def train_model(self, x):
        with tf.variable_scope('train_rnn_dbn'):
            with tf.variable_scope('propagate_states'):
                states = self.__unroll_rnn(x)
                state0 = self.rnn_s0
                if self.num_rnn_cells > 1:
                    states = states[-1]
                    state0 = state0[-1]
                u_t = tf.reshape(states.c, [-1, self.s_size])
                q_t = tf.reshape(states.h, [-1, self.s_size])
                u_tm1 = tf.concat([state0.c, u_t], 0)[:-1, :]
                q_tm1 = tf.concat([state0.h, q_t], 0)[:-1, :]

            # Make an rbm between each layer of the dbn so that
            # we can train each layer greedily
            with tf.variable_scope('make_rbms'):
                rbm_layers = [x]
                rbms = []
                for i in range(len(self.dbn_sizes) - 1):
                    bv = tf.matmul(u_tm1, self.Wu[i]) + tf.matmul(q_tm1, self.Wq[i]) + self.B[i]
                    bh = tf.matmul(u_tm1, self.Wu[i+1]) + tf.matmul(q_tm1, self.Wq[i+1]) + self.B[i+1]
                    rbms.append(RBM(self.W[i], bv, bh))
                    visible_layer = tf.sigmoid(tf.matmul(rbm_layers[-1], self.W[i]) + self.B[i+1])
                    rbm_layers.append(sample(visible_layer))

        # Create a list of optimizers and other subgraphs, one for each rbm
        # that we will train
        with tf.variable_scope('train_ops'):
            costs = []
            optimizers = []
            loglikelihoods = []
            summaries = []
            for i in range(len(rbms)):
                cost, loglikelihood = rbms[i].free_energy_cost(rbm_layers[i], 15)
                cost_summary = tf.summary.scalar('free_energy', cost)
                ll_summary = tf.summary.scalar('log_likelihood', loglikelihood)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                gradients = optimizer.compute_gradients(cost)
                gradients = [(tf.clip_by_value(grad, -10.0, 10.0), var)
                             for grad, var in gradients if grad is not None]
                optimizer = optimizer.apply_gradients(gradients)
                costs.append(cost)
                optimizers.append(optimizer)
                loglikelihoods.append(loglikelihood)
                summaries.append([cost_summary, ll_summary])

        return costs, optimizers, loglikelihoods, summaries

    def pretrain_model(self, x):
        with tf.variable_scope('pre-train_dbn'):

            # Make an rbm between each layer of the dbn so that
            # we can train each layer greedily
            with tf.variable_scope('make_rbms'):
                rbm_layers = [x]
                rbms = []
                for i in range(len(self.dbn_sizes) - 1):
                    rbms.append(RBM(self.W[i], self.B[i], self.B[i+1]))
                    visible_layer = tf.sigmoid(tf.matmul(rbm_layers[-1], self.W[i]) + self.B[i+1])
                    rbm_layers.append(sample(visible_layer))

        # Create a list of optimizers and other subgraphs, one for each rbm
        # that we will train
        with tf.variable_scope('pre-train_ops'):
            costs = []
            optimizers = []
            for i in range(len(rbms)):
                cost, _ = rbms[i].free_energy_cost(rbm_layers[i], 1)
                optimizer = tf.train.AdamOptimizer().minimize(cost)
                costs.append(cost)
                optimizers.append(optimizer)
        return costs, optimizers

    # Propagate the states of the lstm forward in time
    # with a batch of training inputs
    def __unroll_rnn(self, x):
        with tf.variable_scope('unroll_rnn'):
            def recurrence(s_tm1, _x):
                _x = tf.reshape(_x, [1, self.v_size])
                _, s_t = self.rnn(_x, s_tm1)
                return s_t
            states = tf.scan(recurrence, x, initializer=self.rnn_s0)
        return states
