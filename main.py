from MusicGenerator import MusicGenerator
import Helpers as hp


musician = MusicGenerator(hp.Configuration(epochs=200, train_log_path='./jazz', num_timesteps=4))
musician.train('./training_songs/Nottingham')
music = musician.generate(400, './training_songs/Nottingham/ashover_simple_chords_5.mid',
                          './generated_music', 'jazz')
