from MusicGenerator import MusicGenerator
import Helpers as hp


musician = MusicGenerator(hp.Configuration(epochs=1000))
# musician.train('/Users/Eric/ML_data/MusicGenerator')
# music = musician.generate(300, '/Users/Eric/ML_data/MusicGenerator/Every_Time_We_Touch_-_Chorus.mid',
#                           './generated_music', 'test')
musician.train('./training_songs')
music = musician.generate(300, './training_songs/Every_Time_We_Touch_-_Chorus.mid',
                          './generated_music', 'test')
