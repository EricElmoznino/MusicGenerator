from __future__ import print_function
from midi_manipulation.utils import midiread, midiwrite
import numpy as np


class MidiManipulator:
    def __init__(self, num_timesteps, lower_bound=21, upper_bound=109):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.span = upper_bound - lower_bound
        self.num_timesteps = num_timesteps

    @property
    def input_length(self):
        return self.span * self.num_timesteps

    def write_song(self, path, song):
        song = np.reshape(song, (song.shape[0]*self.num_timesteps, self.span))
        midiwrite(path, song, r=(self.lower_bound, self.upper_bound), dt=0.1)

    def get_song(self, path):
        song = np.array(midiread(path, r=(self.lower_bound, self.upper_bound), dt=0.1).piano_roll)
        song = song[:int(song.shape[0]/self.num_timesteps)*self.num_timesteps]
        song = np.reshape(song, [int(song.shape[0]/self.num_timesteps), self.span*self.num_timesteps])
        return song

    def get_songs(self, files, max_size):
        songs = []
        for f in files:
            try:
                song = self.get_song(f)
                for i in range(0, song.shape[0], max_size):
                    songs.append(song[i:i+max_size, :])
            except Exception as e:
                print (f, e)
        return songs
