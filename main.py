from MusicGenerator import MusicGenerator
import Helpers as hp


musician = MusicGenerator(hp.Configuration(epochs=500, train_log_path='./train1m50h100s4t'))
musician.train('./training_songs/Jazz')
music = musician.generate(400, './training_songs/Jazz/005-A_Nighting.mid',
                          './generated_music', '1m50h100s4t')
