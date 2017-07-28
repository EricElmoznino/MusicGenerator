from __future__ import print_function
import tensorflow as tf
import time
import webbrowser
from subprocess import Popen, PIPE
import os


class Configuration:
    def __init__(self, train_log_path = './train', epochs=50, pretrain_epochs=50,
                 batch_size=100, pretrain_batch_size=100, num_timesteps=5,
                 train_sample_iters=15, pretrain_sample_iters=1, gen_sample_iters=25):
        self.train_log_path = train_log_path
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.pretrain_batch_size = pretrain_batch_size
        self.num_timesteps = num_timesteps
        self.train_sample_iters = train_sample_iters
        self.pretrain_sample_iters = pretrain_sample_iters
        self.gen_sample_iters = gen_sample_iters


def files_at_path(path):
    return [os.path.join(path, name) for name in os.listdir(path)]


def log_step(step, total_steps, start_time, error):
    progress = int(step / float(total_steps) * 100)
    seconds = time.time() - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print(str(progress) + '%\t|\t',
          int(h), 'hours,', int(m), 'minutes,', int(s), 'seconds\t|\t',
          'Step:', step, '/', total_steps, '\t|\t',
          'Error:', error)


def log_epoch(epoch, total_epochs, error):
    print('\nEpoch', epoch, 'completed out of', total_epochs,
          ':\tError:', error, '\n')


def weight_variables(shape, name='weights'):
    initial = tf.truncated_normal_initializer(stddev=0.1)
    return tf.get_variable(name, shape=shape,
                           initializer=initial)


def bias_variables(shape, name='biases'):
    initial = tf.constant_initializer(0.1)
    return tf.get_variable(name, shape=shape,
                           initializer=initial)


def open_tensorboard(train_log_path):
    tensorboard = Popen(['tensorboard', '--logdir=' + train_log_path],
                        stdout=PIPE, stderr=PIPE)
    time.sleep(5)
    webbrowser.open('http://0.0.0.0:6006')
    while input('Press <q> to quit') != 'q':
        continue
    tensorboard.terminate()
