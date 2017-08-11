from MusicGenerator import MusicGenerator
import Helpers as hp


musician = MusicGenerator(hp.Configuration(epochs=200, train_log_path='./classicalrnn'))
musician.train('./training_songs/Classical')
music = musician.generate(3200, './training_songs/Classical/017-BEETHOVEN_-_Op-013_No-08_Pathetique_3rd-Mov_1799.mid',
                          './generated_music', 'classicalrnn')
