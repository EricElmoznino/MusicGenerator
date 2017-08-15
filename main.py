from MusicGenerator import MusicGenerator
import Helpers as hp


# musician = MusicGenerator(hp.Configuration(epochs=400, train_log_path='./dbn_long_1pre', pretrain_epochs=1))
# musician.train('./training_songs/Pop')
# music = musician.generate(3200, './training_songs/Pop/I_Kissed_A_Girl_-_Chorus.mid',
#                           './generated_music', 'dbn_long_1pre')

musician = MusicGenerator(hp.Configuration(epochs=200, train_log_path='./nottingham', pretrain_epochs=0))
musician.train('./training_songs/Nottingham')
music = musician.generate(3200, './training_songs/Nottingham/ashover_simple_chords_32.mid',
                          './generated_music', 'nottingham')
