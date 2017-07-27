from __future__ import print_function
import tensorflow as tf
import time
import webbrowser
from subprocess import Popen, PIPE


class Configuration:
    def __init__(self, train_log_path = './train', epochs=10, batch_size=10, dropout=0.0):
        self.train_log_path = train_log_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.keep_prob = 1-dropout


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
          ':\tError:', error)


def log_generic(error, set_name):
    print('Average on', set_name, 'set:\t', 'Error:', error, '\n')


def weight_variables(shape):
    initial = tf.truncated_normal_initializer(stddev=0.1)
    return tf.get_variable('weights', shape=shape,
                           initializer=initial)


def bias_variables(shape):
    initial = tf.constant_initializer(0.1)
    return tf.get_variable('biases', shape=shape,
                           initializer=initial)


def open_tensorboard(train_log_path):
    tensorboard = Popen(['tensorboard', '--logdir=' + train_log_path],
                        stdout=PIPE, stderr=PIPE)
    time.sleep(5)
    webbrowser.open('http://0.0.0.0:6006')
    while input('Press <q> to quit') != 'q':
        continue
    tensorboard.terminate()
