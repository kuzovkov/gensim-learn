#coding=utf-8

class MyCorpus(object):

    dictionary = None
    filename = 'mycorpus.txt'

    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.filename = '/'.join(__file__.split('/')[:-1] + [self.filename])

    def __iter__(self):
        for line in open(self.filename):
            # assume there's one document per line, tokens separated by whitespace
            yield self.dictionary.doc2bow(line.lower().split())