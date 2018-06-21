# MusicGenerator #
A deep belief net coupled with a recurrent LSTM network that composes original piano pieces. Implemented in TensorFlow.
To see some sample generated songs, see the portfolio page of my website https://ericelmoznino.wixsite.com/work/portfolio (AI Musician - 2017)

### Requirements ###
- Python 3
- TensorFlow (confirmed working with version 1.4, but will likely work for ealier and more recent distributions as well)
- A midi file player (to listen to the training data and generated songs)

### Instructions ###
The process for how to train a model and generate music should be clear from looking at the template in 'main.py'. It consists of:
1. Instantiating a `MusicGenerator` object with the desired training and model configurations
2. Calling the `MusicGenerator` object's `train(dataset_path)` method on a folder of midi files
3. Calling the `MusicGenerator` object's `generate(length, primer_song, save_folder, song_name)` method. The method will both save the song at the path and return the generated music

Some datasets have been packaged with the repository to quickly make sure everything is working fine. 

### File Descriptions ###
- MusicGenerator.py:
Contains the principal class for model instantiation, training, and generation. All the session logic is in here. Training data must be music in .mid format, and composed music is in .mid as well.
- RNN_DBN.py:
A deep belief net that is conditioned on the state of a recurrent net. The class is generalized and can work for a variety of appropriate tasks, not only music generation.
- DBN.py:
Class implementation of a deep belief net (stacked RBM's).
- RBM.py:
Class implementation of an RBM.
- Helpers.py:
Convenience functions and classes.
- MidiManipulator.py:
Logic to process .mid files into numpy arrays.
- main.py:
An example of training the model and generating music with it.
