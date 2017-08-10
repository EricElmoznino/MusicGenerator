# -*- coding: ISO-8859-1 -*-

from .RawInstreamFile import RawInstreamFile
from .MidiFileParser import MidiFileParser


class MidiInFile:

    def __init__(self, outStream, infile):
        # these could also have been mixins, would that be better? Nah!
        self.raw_in = RawInstreamFile(infile)
        self.parser = MidiFileParser(self.raw_in, outStream)


    def read(self):
        "Start parsing the file"
        p = self.parser
        p.parseMThdChunk()
        p.parseMTrkChunks()


    def setData(self, data=''):
        "Sets the data from a plain string"
        self.raw_in.setData(data)
    
    
