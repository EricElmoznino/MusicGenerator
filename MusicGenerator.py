import tensorflow as tf
import Helpers as hp
import MidiManipulator as mm
from RNN_RBM import RNN_RBM
import shutil
import os
import time


class MusicGenerator:

    def __init__(self, configuration):
        self.conf = configuration
        self.manipulator = mm.MidiManipulator(self.conf.num_timesteps)

        self.rnn_rbm = RNN_RBM(self.manipulator.input_length, 50, 100)

        with tf.variable_scope('inputs'):
            self.x = tf.placeholder(tf.float32, shape=[None, self.manipulator.input_length])

        self.saver = tf.train.Saver()

    def generate(self, length, primer_song, generation_path, name, primer_length=100):
        primer_song = self.manipulator.get_song(primer_song)[0:primer_length, :]
        model = self.rnn_rbm.generation_model(self.x, length)
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
        cost, optimizer, summary = self.rnn_rbm.train_model(self.x)
        self.__run_updates(train_path, optimizer, cost, [summary], False)

    def __pre_train(self, train_path):
        print('Starting pre-training\n')
        cost, optimizer, summary = self.rnn_rbm.pretrain_model(self.x)
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
            if not pre_training:
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
