
import logging
import re
import pandas as pd
import text_utility
from review import ReviewsDAL
from gensim import corpora,models,matutils
import nltk

PosReview = 'POSREVIEW'
NegReview = 'NEGREVIEW'

class WordsStream(object):
    PosWords = frozenset( pd.read_csv('extend_poswords.txt',header=None).values.flatten() )
    NegWords = frozenset( pd.read_csv('extend_negwords.txt',header=None).values.flatten() )

    def append_sentiment_words(self,words):
        extra_words = []
        for w in words:
            if w in WordsStream.PosWords:
                extra_words.append(PosReview)
                # print "+++ {}: Positive".format(w)

            if w in WordsStream.NegWords:
                extra_words.append(NegReview)
                # print "--- {}: Negative".format(w)

        words += extra_words

    def stream(self):
        dal = ReviewsDAL()
        review_stream = dal.load_words()
        for index, r in enumerate(review_stream):

            self.append_sentiment_words(r.sent.words)
            yield r.sent.words

            if index % 300 == 0:
                print "{} examples loaded from mongodb".format(index + 1)

        dal.close()

def build_dictionary():
    words_stream = WordsStream().stream()

    dictionary = corpora.Dictionary(words_stream)
    dictionary.save('codeword_dictionary.dict')  # store the dictionary, for future reference
    dictionary.save_as_text("codeword_dictionary.txt", sort_by_word=False)

    print "======== Dictionary Generated and Saved ========"

def load_extra_stopwords():
    extra_stop_words = set()
    with open('extra_stopwords.txt','rt') as inf:
        for line in inf:
            w = line.strip()
            if len(w) >= 1:
                extra_stop_words.add(w)
                extra_stop_words.add(w + text_utility.NEG_SUFFIX)
    return extra_stop_words

class DictCleaner(object):
    def __init__(self):
        self._invalid_pattern = re.compile(r"[^a-zA-Z_\-]")
        self._extra_stop_words = load_extra_stopwords()

    def is_token_invalid(self,token):
        if self._invalid_pattern.search(token) is not None:
            return True

        if token in self._extra_stop_words:
            return True

        return False

    def clean(self,no_below=5, keep_n=100000):
        dictionary = corpora.Dictionary.load('codeword_dictionary.dict')
        print "originally, there are {} tokens".format(len(dictionary))

        # !!! we cannot filter out 'too frequent'
        # !!! because 'POSREVIEW' and 'NEGREVIEW' are just two most frequent words
        dictionary.filter_extremes(no_below=no_below,no_above=1,keep_n=keep_n)
        print "after filtering too rare, there are {} tokens".format(len(dictionary))

        # filter out invalid tokens
        invalid_tokenids = [id for token,id in dictionary.token2id.viewitems()
                            if self.is_token_invalid(token) ]
        print "there are {} tokens are invalid".format(len(invalid_tokenids))

        dictionary.filter_tokens(bad_ids = invalid_tokenids)
        print "after filtering invalid, there are {} tokens".format(len(dictionary))

        return dictionary

def clean_dict_save():
    cleaner = DictCleaner()
    clean_dict = cleaner.clean(no_below=10)

    clean_dict.save('codeword_dictionary.dict')
    # sort by decreasing doc-frequency
    clean_dict.save_as_text("codeword_dictionary.txt", sort_by_word=False)

    print "##################### dictionary is cleaned, shrinked and saved"

def build_bow_save():
    dictionary = corpora.Dictionary.load('codeword_dictionary.dict')

    print "\n=========== begin BOW, ......"
    wstream = WordsStream()
    bow_stream = (dictionary.doc2bow(words) for words in wstream.stream())

    target_file = "codeword_airbnb.bow"
    corpora.MmCorpus.serialize(target_file, bow_stream)
    print "##################### BOW saved #####################"

def print_topic_distribution(model,filename):
    # ----------- print a template waiting for editing
    with open(filename + ".csv", "wt") as outf:
        outf.write("id,name,distribution\n")
        topics = model.show_topics(num_topics=-1, log=False, formatted=True)
        for topic in topics:
            # leave blank in the middle, waiting for naming
            outf.write('{},Undef{},{}\n'.format(topic[0],topic[0],topic[1]))

    # ----------- print a full version for human reading
    with open(filename+"_full.txt","wt") as outf:
        # ---------- write each topic and words' contribution
        topics = model.show_topics(num_topics=-1, log=False, formatted=True)
        for topic in topics:
            # topic[0]: topic number
            # topic[1]: topic description
            outf.write("\n############# TOPIC {} #############\n".format(topic[0]))
            outf.write(topic[1]+"\n")

        # ---------- words statistics in all topics
        outf.write("\n\n\n****************** KEY WORDS ******************\n")
        topics = model.show_topics(num_topics=-1, log=False, formatted=False)
        keywords = (word for (_,words) in topics for (word,score) in words)

        fdist = nltk.FreqDist(keywords)
        for index,(w,c) in enumerate( fdist.most_common(100) ):
            outf.write("{}-th keyword: <{},{}>\n".format(index+1,w,c))

def run_lda(n_topics,n_pass=1,parallel=False):
    dictionary = corpora.Dictionary.load("codeword_dictionary.dict")
    bow = corpora.MmCorpus('codeword_airbnb.bow')

    if parallel:
        model = models.LdaMulticore(bow, id2word=dictionary, num_topics=n_topics,passes=n_pass)
    else:
        model = models.LdaModel(bow, id2word=dictionary, num_topics=n_topics,passes=n_pass)
    print "##################### LDA has been built #####################"

    # --------------- save result
    model.save('airbnb.lda_model')
    print_topic_distribution(model,'airbnb_topics')


def load_and_print_topics():
    model = models.LdaModel.load("airbnb.lda_model")
    print_topic_distribution(model,'airbnb_topics')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # build_dictionary()

    # clean_dict_save()
    # build_bow_save()
    run_lda(100)



