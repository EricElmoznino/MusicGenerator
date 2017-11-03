# MusicGenerator
A deep belief net coupled with a recurrent LSTM network that composes original piano pieces.

File Descriptions:
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
