from MusicGenerator import MusicGenerator
import Helpers as hp


musician = MusicGenerator(hp.Configuration(epochs=200))
# musician.train('/Users/Eric/ML_data/MusicGenerator')
# music = musician.generate(300, '/Users/Eric/ML_data/MusicGenerator/Every_Time_We_Touch_-_Chorus.mid',
#                           './generated_music', 'test')
musician.train('./training_songs')
music = musician.generate(300, './training_songs/Rolling_In_The_Deep_-_Chorus.mid',
                          './generated_music', 'test')
