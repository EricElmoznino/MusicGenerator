from MusicGenerator import MusicGenerator
import Helpers as hp


musician = MusicGenerator(hp.Configuration(epochs=300, train_log_path='./jasslstm4m4t', num_timesteps=4))
musician.train('./training_songs/Jazz')
music = musician.generate(3200, './training_songs/Jazz/005-A_Nighting.mid',
                          './generated_music', 'jasslstm4m4t')
