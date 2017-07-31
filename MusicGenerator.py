import tensorflow as tf
import Helpers as hp
import MidiManipulator as mm
from RBM import RBM
import shutil
import os
import time


class MusicGenerator:

    def __init__(self, configuration):
        self.conf = configuration
        self.manipulator = mm.MidiManipulator(self.conf.num_timesteps)

        self.rbm_visible_size = self.manipulator.input_length
        self.rbm_hidden_size = 50
        self.rnn_state_size = 100

        with tf.variable_scope('inputs'):
            self.x = tf.placeholder(tf.float32, shape=[None, self.manipulator.input_length])

        with tf.variable_scope('rbm'):
            self.w_rbm = hp.weight_variables([self.rbm_visible_size, self.rbm_hidden_size],
                                             stddev=0.01)

        with tf.variable_scope('rnn_to_rbm'):
            self.w_rnn_to_rbm_h = hp.weight_variables([self.rnn_state_size, self.rbm_hidden_size],
                                                      stddev=0.01, name='weights_hidden')
            self.w_rnn_to_rbm_v = hp.weight_variables([self.rnn_state_size, self.rbm_visible_size],
                                                      stddev=0.01, name='weights_visible')
            self.b_rnn_to_rbm_h = hp.bias_variables([self.rbm_hidden_size],
                                                    value=0.0, name='biases_hidden')
            self.b_rnn_to_rbm_v = hp.bias_variables([self.rbm_visible_size],
                                                    value=0.0, name='biases_visible')

        with tf.variable_scope('rnn'):
            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_state_size)

        self.saver = tf.train.Saver()

    def generate(self, length, primer_song, generation_path, name, primer_length=100):
        primer_song = self.manipulator.get_song(primer_song)[0:primer_length, :]
        primer_length = primer_song.shape[0]
        model = self.__build_generation_model(length, primer_length)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, os.path.join(self.conf.train_log_path, 'model.ckpt'))
            music = sess.run(model, feed_dict={self.x: primer_song})
        self.manipulator.write_song(os.path.join(generation_path, name+'.mid'), music)
        return music

    def train(self, train_path):
        if os.path.exists(self.conf.train_log_path):
            shutil.rmtree(self.conf.train_log_path)
        os.mkdir(self.conf.train_log_path)

        self.__pre_train(train_path)

        print('Starting training\n')
        cost, optimizer, summary = self.__build_train_model()
        self.__run_updates(train_path, optimizer, cost, [summary], False)

    def __pre_train(self, train_path):
        print('Starting pre-training\n')
        cost, optimizer, summary = self.__build_pretrain_model()
        self.__run_updates(train_path, optimizer, cost, [summary], True)

    def __run_updates(self, train_path, optimizer, cost, summaries, pre_training):
        if pre_training:
            epochs = self.conf.pretrain_epochs
            batch_size = self.conf.pretrain_batch_size
            model_name = 'pre_train'
        else:
            epochs = self.conf.epochs
            batch_size = self.conf.batch_size
            model_name = 'model'

        summaries = tf.summary.merge(summaries)

        songs = self.manipulator.get_songs(hp.files_at_path(train_path), batch_size)
        n_batches = len(songs)
        n_steps = n_batches * epochs

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if pre_training:
                self.__initialize_bv(sess, songs)
            else:
                self.saver.restore(sess, os.path.join(self.conf.train_log_path, 'pre_train.ckpt'))

            train_writer = tf.summary.FileWriter(self.conf.train_log_path, sess.graph)

            start_time = time.time()
            step = 0
            for epoch in range(1, epochs + 1):
                epoch_cost = 0
                for batch in range(n_batches):
                    song_batch = songs[batch]

                    if step % max(int(n_steps / 1000), 1) == 0:
                        _, c, s = sess.run([optimizer, cost, summaries],
                                           feed_dict={self.x: song_batch})
                        train_writer.add_summary(s, step)
                        hp.log_step(step, n_steps, start_time, c)
                    else:
                        _, c = sess.run([optimizer, cost],
                                        feed_dict={self.x: song_batch})

                    epoch_cost += c
                    step += 1

                hp.log_epoch(epoch, self.conf.epochs, epoch_cost / n_batches)

            self.saver.save(sess, os.path.join(self.conf.train_log_path, model_name+'.ckpt'))

    def __build_generation_model(self, length, primer_length):
        with tf.variable_scope('generation'):
            with tf.variable_scope('primer'):
                primer_state = self.rnn_cell.zero_state(1, tf.float32)
                for i in range(primer_length):
                    _, primer_state = self.rnn_cell(tf.reshape(self.x[i, :], [1, -1]),
                                                    primer_state)

            def music_timestep(t, k, x_t, state_tm1, music):
                rnn_state = tf.reshape(state_tm1.c, [1, self.rnn_state_size])
                bh = tf.matmul(rnn_state, self.w_rnn_to_rbm_h) + self.b_rnn_to_rbm_h
                bv = tf.matmul(rnn_state, self.w_rnn_to_rbm_v) + self.b_rnn_to_rbm_v
                rbm = RBM(self.w_rbm, bv, bh)
                note = rbm.gibbs_sample(x_t, self.conf.gen_sample_iters, trainable=False)
                _, state_t = self.rnn_cell(note, state_tm1)
                music = music + tf.concat([tf.zeros([t, self.manipulator.input_length]), note,
                                           tf.zeros([k-t-1, self.manipulator.input_length])], 0)
                return t+1, k, note, state_t, music

            count = tf.constant(0)
            music = tf.zeros([length, self.manipulator.input_length], tf.float32)
            _, _, _, _, music = tf.while_loop(lambda t, k, *args: t < k, music_timestep,
                                              [count, length,
                                               tf.zeros([1, self.manipulator.input_length], tf.float32),
                                               primer_state, music],
                                              back_prop=False)
        return music

    def __build_train_model(self):
        with tf.variable_scope('train_rnn_rbm'):
            def unroll_rnn(state_tm1, x_t):
                x_t = tf.reshape(x_t, [1, -1])
                _, state_t = self.rnn_cell(x_t, state_tm1)
                return state_t

            states = tf.scan(unroll_rnn, self.x,
                             initializer=self.rnn_cell.zero_state(1, tf.float32))
            states = tf.reshape(states.c, [-1, self.rnn_state_size])

            bh = tf.matmul(states, self.w_rnn_to_rbm_h) + self.b_rnn_to_rbm_h
            bv = tf.matmul(states, self.w_rnn_to_rbm_v) + self.b_rnn_to_rbm_v
            rbm = RBM(self.w_rbm, bv, bh)

        with tf.variable_scope('train_ops'):
            cost = rbm.free_energy_cost(self.x, self.conf.train_sample_iters)
            cost_summary = tf.summary.scalar('pre-train_cost', cost)
            optimizer = tf.train.AdamOptimizer().minimize(cost)

        return cost, optimizer, cost_summary

    def __build_pretrain_model(self):
        with tf.variable_scope('pre-train_rbm'):
            rbm = RBM(self.w_rbm, self.b_rnn_to_rbm_v, self.b_rnn_to_rbm_h)

        with tf.variable_scope('pre-train_ops'):
            cost = rbm.free_energy_cost(self.x, self.conf.pretrain_sample_iters)
            cost_summary = tf.summary.scalar('pre-train_cost', cost)
            optimizer = tf.train.AdamOptimizer().minimize(cost)

        return cost, optimizer, cost_summary

    def __initialize_bv(self, sess, songs):
        data = tf.constant(0.0, shape=[0, self.manipulator.input_length])
        for song in songs:
            data = tf.concat((data, song), 0)
        avg = tf.reduce_mean(data, axis=0)
        sess.run(tf.assign(self.b_rnn_to_rbm_v, avg))
