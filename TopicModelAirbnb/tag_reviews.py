
import pandas as pd
from gensim import corpora,models,matutils
from sentence import Sentence
from review import Review,ReviewsDAL
from word_coder import WordCoder
import text_utility
import nltk

class MixTopic(object):
    SentimentMapping = {'+':1,'-':-1,'?':0}

    def __init__(self,text):
        """
        text must be in the format like <sentiment>_<tag1>_<tag2>_...
        """
        segments = text.split('_')
        self.sentiment = MixTopic.SentimentMapping[ segments[0] ]
        self.tags = pd.Series( {k:1 for k in segments[1:]}  )

    def weight(self, other):
        weight = float(other)
        self.tags *= weight
        self.sentiment *= weight
        return self

    def add(self, other):
        self.sentiment += other.sentiment
        self.tags = self.tags.add(other.tags,fill_value=0)
        return self

    def normalize(self):
        self.tags /= self.tags.sum()
        self.tags.sort_values(ascending=False)

    def __str__(self):
        return str(self.tags)

stop_words = text_utility.make_stop_words()
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

lda_model = models.LdaModel.load("airbnb.lda_model")
dictionary = corpora.Dictionary.load('codeword_dictionary.dict')
wordcoder = WordCoder()
topic_mapping = pd.read_csv('airbnb_topics.csv',index_col='id').iloc[:,0].to_dict()

def print_topics(txt):
    sentence = Sentence.from_raw(txt,stop_words)
    print "\n{}\n".format(sentence.raw)

    coded_words = wordcoder.code(sentence.words)
    bow = dictionary.doc2bow(coded_words)

    topic_distribution = lda_model[bow]
    topic_distribution.sort(key=lambda t: t[1], reverse=True)

    tags = None
    for index, (topic_id, topic_percentage) in enumerate(topic_distribution):
        mt = MixTopic(topic_mapping[topic_id])
        mt.weight(topic_percentage)

        if tags is None:
            tags = mt
        else:
            tags.add(mt)

    tags.normalize()
    print tags

if __name__ == "__main__":
    dal = ReviewsDAL()
    review_stream = dal.sampling(10)

    for index,review in enumerate( review_stream):
        print "*********** [{}] ***********".format(index+1)

        for sentence in sent_tokenizer.tokenize(review.sent.raw):
            print_topics(sentence)

    dal.close()
