from __future__ import print_function
import midi
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
        # return 2 * self.span * self.num_timesteps
        return self.span * self.num_timesteps


    def write_song(self, path, song):
        #Reshape the song into a format that midi_manipulation can understand, and then write the song to disk
        # song = np.reshape(song, (song.shape[0]*self.num_timesteps, 2*self.span))
        # self.note_state_matrix_to_midi(song, path)
        song = np.reshape(song, (song.shape[0]*self.num_timesteps, self.span))
        midiwrite(path, song, r=(self.lower_bound, self.upper_bound))

    def get_song(self, path):
        #Load the song and reshape it to place multiple timesteps next to each other
        # song = np.array(self.midi_to_note_state_matrix(path))
        song = np.array(midiread(path, r=(self.lower_bound, self.upper_bound)).piano_roll)
        song = song[:int(song.shape[0]/self.num_timesteps)*self.num_timesteps]
        # song = np.reshape(song, [int(song.shape[0]/self.num_timesteps), 2*self.span*self.num_timesteps])
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

    def midi_to_note_state_matrix(self, midifile):
        pattern = midi.read_midifile(midifile)

        timeleft = [track[0].tick for track in pattern]

        posns = [0 for track in pattern]

        state_matrix = []
        time = 0

        state = [[0,0] for x in range(self.span)]
        state_matrix.append(state)
        condition = True
        while condition:
            if time % (pattern.resolution / 4) == (pattern.resolution / 8):
                # Crossed a note boundary. Create a new state, defaulting to holding notes
                oldstate = state
                state = [[oldstate[x][0],0] for x in range(self.span)]
                state_matrix.append(state)
            for i in range(len(timeleft)): #For each track
                if not condition:
                    break
                while timeleft[i] == 0:
                    track = pattern[i]
                    pos = posns[i]

                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch < self.lower_bound) or (evt.pitch >= self.upper_bound):
                            pass
                            # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                        else:
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch-self.lower_bound] = [0, 0]
                            else:
                                state[evt.pitch-self.lower_bound] = [1, 1]
                    elif isinstance(evt, midi.TimeSignatureEvent):
                        if evt.numerator not in (2, 4):
                            # We don't want to worry about non-4 time signatures. Bail early!
                            # print "Found time signature event {}. Bailing!".format(evt)
                            out =  state_matrix
                            condition = False
                            break
                    try:
                        timeleft[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        timeleft[i] = None

                if timeleft[i] is not None:
                    timeleft[i] -= 1

            if all(t is None for t in timeleft):
                break

            time += 1

        S = np.array(state_matrix)
        state_matrix = np.hstack((S[:, :, 0], S[:, :, 1]))
        state_matrix = np.asarray(state_matrix).tolist()
        return state_matrix

    def note_state_matrix_to_midi(self, state_matrix, path):
        state_matrix = np.array(state_matrix)
        if not len(state_matrix.shape) == 3:
            state_matrix = np.dstack((state_matrix[:, :self.span], state_matrix[:, self.span:]))
        state_matrix = np.asarray(state_matrix)
        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)

        span = self.upper_bound - self.lower_bound
        tickscale = 55

        lastcmdtime = 0
        prevstate = [[0,0] for x in range(span)]
        for time, state in enumerate(state_matrix + [prevstate[:]]):
            offNotes = []
            onNotes = []
            for i in range(span):
                n = state[i]
                p = prevstate[i]
                if p[0] == 1:
                    if n[0] == 0:
                        offNotes.append(i)
                    elif n[1] == 1:
                        offNotes.append(i)
                        onNotes.append(i)
                elif n[0] == 1:
                    onNotes.append(i)
            for note in offNotes:
                track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale,
                                               pitch=note+self.lower_bound))
                lastcmdtime = time
            for note in onNotes:
                track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale,
                                              velocity=40, pitch=note+self.lower_bound))
                lastcmdtime = time

            prevstate = state

        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)

        midi.write_midifile(path, pattern)
