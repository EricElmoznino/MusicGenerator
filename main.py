from MusicGenerator import MusicGenerator
import Helpers as hp


musician = MusicGenerator(hp.Configuration(epochs=400, train_log_path='./dbn4m', pretrain_epochs=100))
musician.train('./training_songs/Pop')
music = musician.generate(3200, './training_songs/Pop/I_Kissed_A_Girl_-_Chorus.mid',
                          './generated_music', 'dbn4m')
