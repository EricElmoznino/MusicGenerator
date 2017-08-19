from MusicGenerator import MusicGenerator
import Helpers as hp


# musician = MusicGenerator(hp.Configuration(epochs=200, train_log_path='./dbn'))
# musician.train('./training_songs/Pop')
# music = musician.generate(3200, './training_songs/Pop/I_Kissed_A_Girl_-_Chorus.mid',
#                           './generated_music', 'dbn')
musician = MusicGenerator(hp.Configuration(epochs=200, train_log_path='./lstm_theano_comp_jazz'))
musician.train('./training_songs/Jazz')
music = musician.generate(3200, './training_songs/Jazz/005-A_Nighting.mid',
                          './generated_music', 'lstm_theano_comp_jazz')
