import tensorflow as tf
import Helpers as hp


class LSTM:

    def __init__(self, in_size, s_size):
        self.in_size = in_size
        self.s_size = s_size

        with tf.variable_scope('lstm_cell'):
            self.Wvu = hp.weight_variables([in_size, s_size], 0.0001, name='Wvu')
            self.Wuu = hp.weight_variables([s_size, s_size], 0.0001, name='Wuu' )
            self.Bu = hp.bias_variables([s_size], 0.0, name='Bu')
            self.Wui = hp.weight_variables([s_size, s_size], 0.0001, name='Wui')
            self.Wqi = hp.weight_variables([s_size, s_size], 0.0001, name='Wqi')
            self.Wci = hp.weight_variables([s_size, s_size], 0.0001, name='Wci')
            self.Bi = hp.bias_variables([s_size], 0.0, name='Bi')
            self.Wuf = hp.weight_variables([s_size, s_size], 0.0001, name='Wuf')
            self.Wqf = hp.weight_variables([s_size, s_size], 0.0001, name='Wqf')
            self.Wcf = hp.weight_variables([s_size, s_size], 0.0001, name='Wcf')
            self.Bf = hp.bias_variables([s_size], 0.0, name='Bf')
            self.Wuc = hp.weight_variables([s_size, s_size], 0.0001, name='Wuc')
            self.Wqc = hp.weight_variables([s_size, s_size], 0.0001, name='Wqc')
            self.Bc = hp.bias_variables([s_size], 0.0, name='Bc')
            self.Wuo = hp.weight_variables([s_size, s_size], 0.0001, name='Wuo')
            self.Wqo = hp.weight_variables([s_size, s_size], 0.0001, name='Wqo')
            self.Wco = hp.weight_variables([s_size, s_size], 0.0001, name='Wco')
            self.Bo = hp.bias_variables([s_size], 0.0, name='Bo')

            self.u0 = tf.get_variable('u0', shape=[1, s_size], initializer=tf.zeros_initializer())
            self.q0 = tf.get_variable('q0', shape=[1, s_size], initializer=tf.zeros_initializer())
            self.c0 = tf.get_variable('c0', shape=[1, s_size], initializer=tf.zeros_initializer())

    def step(self, input, s_tm1):
        u_tm1, q_tm1, c_tm1 = s_tm1

        u_t = tf.tanh(self.Bu + tf.matmul(input, self.Wvu) + tf.matmul(u_tm1, self.Wuu))
        i_t = tf.tanh(self.Bi + tf.matmul(c_tm1, self.Wci) + tf.matmul(q_tm1, self.Wqi) + tf.matmul(u_t, self.Wui))
        f_t = tf.tanh(self.Bf + tf.matmul(c_tm1, self.cf), tf.matmul(u_tm1, self.Wuf))
        c_t = tf.multiply(f_t, c_tm1) + tf.multiply(i_t, tf.tanh(
            tf.matmul(u_t, self.Wuc) + tf.matmul(q_tm1, self.Wqc) + self.Bc))
        o_t = tf.tanh(self.Bo + tf.matmul(c_t, self.Wco) + tf.matmul(q_tm1, self.Wqo) + tf.matmul(u_t, self.Wuo))
        q_t = tf.multiply(o_t, tf.tanh(c_t))

        return u_t, q_t, c_t

    def unroll(self, inputs):
        def recurrence(s_tm1, v):
            u_tm1, q_tm1, c_tm1 = s_tm1
            v = tf.reshape(v, [1, -1])

            u_t = tf.tanh(self.Bu + tf.matmul(v, self.Wvu) + tf.matmul(u_tm1, self.Wuu))
            i_t = tf.tanh(self.Bi + tf.matmul(c_tm1, self.Wci) + tf.matmul(q_tm1, self.Wqi) + tf.matmul(u_t, self.Wui))
            f_t = tf.tanh(self.Bf + tf.matmul(c_tm1, self.Wcf) + tf.matmul(u_tm1, self.Wuf))
            c_t = tf.multiply(f_t, c_tm1) + tf.multiply(i_t, tf.tanh(tf.matmul(u_t, self.Wuc) + tf.matmul(q_tm1, self.Wqc) + self.Bc))
            o_t = tf.tanh(self.Bo + tf.matmul(c_t, self.Wco) + tf.matmul(q_tm1, self.Wqo) + tf.matmul(u_t, self.Wuo))
            q_t = tf.multiply(o_t, tf.tanh(c_t))

            return u_t, q_t, c_t

        states = tf.scan(recurrence, inputs, initializer=(self.u0, self.q0, self.c0))
        states = [tf.reshape(s, [-1, self.s_size]) for s in states]
        return states
