from MusicGenerator import MusicGenerator
import Helpers as hp


musician = MusicGenerator(hp.Configuration(epochs=200, train_log_path='./train'))
musician.train('./training_songs/Jazz')
music = musician.generate(400, './training_songs/Jazz/005-A_Nighting.mid',
                          './generated_music', 'test')
