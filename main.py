from MusicGenerator import MusicGenerator
import Helpers as hp


musician = MusicGenerator(hp.Configuration(epochs=200, train_log_path='./lstm_no_theano_pop', pretrain_epochs=0))
musician.train('./training_songs/Pop')
music = musician.generate(3200, './training_songs/Pop/I_Kissed_A_Girl_-_Chorus.mid',
                          './generated_music', 'lstm_no_theano_pop')
