import tensorflow as tf
import Helpers as hp
import MidiManipulator as mm
from RBM import RBM
import numpy as np
import shutil
import os
import time


class MusicGenerator:

    def __init__(self, configuration):
        self.conf = configuration
        self.manipulator = mm.MidiManipulator(self.conf.num_timesteps)

        rbm_visible_size = self.manipulator.input_length()
        rbm_hidden_size = 50
        rnn_state_size = 100

        with tf.variable_scope('rbm'):
            self.w_rbm = hp.weight_variables([rbm_visible_size, rbm_hidden_size], stddev=0.01)

        with tf.variable_scope('rnn_to_rbm'):
            self.w_rnn_to_rbm_h = hp.weight_variables([rnn_state_size, rbm_hidden_size],
                                                      stddev=0.01, name='weights_hidden')
            self.w_rnn_to_rbm_v = hp.weight_variables([rnn_state_size, rbm_visible_size],
                                                      stddev=0.01, name='weights_visible')
            self.b_rnn_to_rbm_h = hp.bias_variables([rbm_hidden_size],
                                                    value=0.0, name='biases_hidden')
            self.b_rnn_to_rbm_v = hp.bias_variables([rbm_visible_size],
                                                    value=0.0, name='biases_visible')

        with tf.variable_scope('rnn'):
            self.cell = tf.contrib.rnn.BasicLSTMCell(rnn_state_size)
            self.cell_initial_state = tf.Variable(tf.zeros([1, rnn_state_size]),
                                                  name='initial_state')

        self.saver = tf.train.Saver()

    def pre_train(self, train_path):
        with tf.variable_scope('pre-train_rbm'):
            x = tf.placeholder(tf.float32, shape=[None, self.manipulator.input_length()])
            rbm = RBM(self.w_rbm, self.b_rnn_to_rbm_v, self.b_rnn_to_rbm_h)

        with tf.variable_scope('pre-train_ops'):
            cost = rbm.free_energy_cost(x, self.conf.pretrain_sample_iters)
            cost_summary = tf.summary.scalar('pre-train_cost', cost)
            optimizer = tf.train.AdamOptimizer().minimize(cost)

        summaries = tf.summary.merge([cost_summary])
        if os.path.exists(self.conf.train_log_path):
            shutil.rmtree(self.conf.train_log_path)
        os.mkdir(self.conf.train_log_path)


        print('Starting pre-training\n')
        songs = self.manipulator.get_songs(hp.files_at_path(train_path), self.conf.pretrain_batch_size)
        n_batches = len(songs)
        n_steps = n_batches * self.conf.pretrain_epochs
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(self.conf.train_log_path, sess.graph)

            start_time = time.time()
            step = 0
            for epoch in range(1, self.conf.epochs + 1):
                epoch_cost = 0
                for batch in range(n_batches):
                    song_batch = songs[batch]

                    if step % max(int(n_steps / 1000), 1) == 0:
                        _, c, s = sess.run([optimizer, cost, summaries],
                                           feed_dict={x: song_batch})
                        train_writer.add_summary(s, step)
                        hp.log_step(step, n_steps, start_time, c)
                    else:
                        _, c = sess.run([optimizer, cost],
                                        feed_dict={x: song_batch})

                    epoch_cost += c
                    step += 1

                hp.log_epoch(epoch, self.conf.epochs, epoch_cost / n_batches)

            self.saver.save(sess, os.path.join(self.conf.train_log_path, 'pre_train.ckpt'))

            # test generation
            sample = rbm.gibbs_sample(x, 1, trainable=False).eval(
                session=sess,
                feed_dict={x: np.zeros((10, self.manipulator.input_length()))}
            )
            for i in range(sample.shape[0]):
                if not any(sample[i,:]):
                    continue
                s = np.reshape(sample[i,:], (self.manipulator.num_timesteps,
                                             2*self.manipulator.span))
                self.manipulator.note_state_matrix_to_midi(s, "generated_chord_{}".format(i))





