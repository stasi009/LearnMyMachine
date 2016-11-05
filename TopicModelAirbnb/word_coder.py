
import re
import csv
import numpy as np
import pandas as pd
import nltk
import text_utility

def extend_sentiment_words_save():
    # since we ignore header, so we only skip 34 rows, not 35 rows
    poswords = pd.read_csv('datas/positive-words.txt',skiprows=34, header=None).values.flatten().tolist()
    print 'totally {} positive words'.format(len(poswords))

    negwords = pd.read_csv('datas/negative-words.txt',skiprows=34, header=None).values.flatten().tolist()
    print "totally {} negative words".format(len(negwords))

    neg_poswords = [w+text_utility.NEG_SUFFIX for w in poswords]
    neg_negwords = [w+text_utility.NEG_SUFFIX for w in negwords]

    poswords += neg_negwords
    negwords += neg_poswords

    pd.Series(poswords).to_csv("datas/extend_poswords.txt",index=False)
    pd.Series(negwords).to_csv("datas/extend_negwords.txt",index=False)

def extract_hostnames(fname):
    reserve_words = set(['and','family','room','private','bed','or','cool',
                         'miles','food','walk','super','house','host','team','space'])
    host_names = set()

    with open(fname,'rt') as inf:
        inf.readline()# read out header

        reader = csv.reader(inf)
        for index,segments in enumerate( reader):
            hostname = segments[3].lower()

            words = re.sub(r" and |&|,|\(|\)|-|\.|\+",' ',hostname)
            words = nltk.word_tokenize(words)

            for word in words:
                if len(word)>0 and word not in reserve_words:
                    host_names.add(word)

    all_host_names = list(host_names)
    all_host_names += [n+text_utility.NEG_SUFFIX for n in host_names]
    all_host_names.sort()

    with open('host_names.txt',"wt") as outf:
        for name in all_host_names:
            outf.write(name+'\n')

def load_extra_stopwords():
    extra_stop_words = set()
    with open('extra_stopwords.txt','rt') as inf:
        for line in inf:
            w = line.strip()
            if len(w) >= 1:
                extra_stop_words.add(w)
                extra_stop_words.add(w + text_utility.NEG_SUFFIX)
    return extra_stop_words

PosReview = 'POSREVIEW'
NegReview = 'NEGREVIEW'
HostName = 'HOSTNAME'

class WordCoder(object):
    def __init__(self):
        self._invalid_pattern = re.compile(r"[^a-zA-Z_\-]")
        self._extra_stop_words = load_extra_stopwords()

        self._poswords = frozenset(pd.read_csv('extend_poswords.txt', header=None).values.flatten())

        self._negwords = set(pd.read_csv('extend_negwords.txt', header=None).values.flatten())
        # ok isn't a negative word in normal sense
        # but it represent a negative feelings in 'review' corpus
        self._negwords.add('ok')

        self._hostnames = set()
        with open("host_names.txt", 'rt') as inf:
            for line in inf:
                self._hostnames.add(line.strip())

    def code(self,words):
        new_words = []
        for word in words:
            if self._invalid_pattern.search(word) is not None:
                continue

            if word in self._extra_stop_words:
                continue

            if word in self._hostnames:
                new_words.append(HostName)# replace with general 'HOSTNAME'
                continue

            new_words.append(word)

            if word in self._poswords:
                new_words.append(PosReview)

            if word in self._negwords:
                new_words.append(NegReview)

        return new_words

if __name__ == "__main__":
    extract_hostnames('listings_summary.csv')
