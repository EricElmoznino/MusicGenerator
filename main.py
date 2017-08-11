from MusicGenerator import MusicGenerator
import Helpers as hp


musician = MusicGenerator(hp.Configuration(epochs=200, train_log_path='./rnn'))
musician.train('./training_songs/Nottingham')
music = musician.generate(3200, './training_songs/Nottingham/ashover_simple_chords_16.mid',
                          './generated_music', 'rnn')
