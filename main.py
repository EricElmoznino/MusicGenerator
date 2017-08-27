from MusicGenerator import MusicGenerator
import Helpers as hp


# musician = MusicGenerator(hp.Configuration(epochs=300, train_log_path='./pop100h2m'))
# musician.train('./training_songs/Pop')
# music = musician.generate(3200, './training_songs/Pop/I_Kissed_A_Girl_-_Chorus.mid',
#                           './generated_music', 'pop100h2m')
musician = MusicGenerator(hp.Configuration(epochs=300, train_log_path='./jazz100h'))
musician.train('./training_songs/Jazz')
music = musician.generate(3200, './training_songs/Jazz/005-A_Nighting.mid',
                          './generated_music', 'jazz100h')
# musician = MusicGenerator(hp.Configuration(epochs=200, train_log_path='./nottingham', pretrain_epochs=0))
# musician.train('./training_songs/Nottingham')
# music = musician.generate(3200, './training_songs/Nottingham/ashover_simple_chords_5.mid',
#                           './generated_music', 'nottingham')
