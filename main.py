from MusicGenerator import MusicGenerator
import Helpers as hp


musician = MusicGenerator(hp.Configuration(epochs=500, train_log_path='./train1m75h100s4t'))
# musician.train('/Users/Eric/ML_data/MusicGenerator/Pop')
# music = musician.generate(300, '/Users/Eric/ML_data/MusicGenerator/Pop/Every_Time_We_Touch_-_Chorus.mid',
#                           './generated_music', 'test')
musician.train('./training_songs/Classical')
music = musician.generate(400, './training_songs/Classical/043-BEETHOVEN_-_Op-031_No-02_Piano_Sonata_No-17_D-min_1st-Mov_Tempest_1802_(pb_Tuncali).mid',
                          './generated_music', '1m75h100s4t')
