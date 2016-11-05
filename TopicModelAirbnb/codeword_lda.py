
import logging
from review import ReviewsDAL
from gensim import corpora,models,matutils
import nltk
from word_coder import WordCoder

def words_stream():
    word_coder = WordCoder()

    dal = ReviewsDAL()
    review_stream = dal.load_words()
    for index, r in enumerate(review_stream):
        yield word_coder.code(r.sent.words)

        if index % 300 == 0:
            print "{} examples loaded from mongodb".format(index + 1)

    dal.close()

def build_dictionary():
    wstream = words_stream()

    dictionary = corpora.Dictionary(wstream)
    dictionary.save('codeword_dictionary.dict')  # store the dictionary, for future reference
    dictionary.save_as_text("codeword_dictionary.txt", sort_by_word=False)

    print "======== Dictionary Generated and Saved ========"

def clean_dict_save(no_below=10, keep_n=100000):
    dictionary = corpora.Dictionary.load('codeword_dictionary.dict')
    print "originally, there are {} tokens".format(len(dictionary))

    # !!! we cannot filter out 'too frequent'
    # !!! because 'POSREVIEW' and 'NEGREVIEW' are just two most frequent words
    dictionary.filter_extremes(no_below=no_below, no_above=1, keep_n=keep_n)
    print "after filtering too rare, there are {} tokens".format(len(dictionary))

    dictionary.save('codeword_dictionary.dict')
    # sort by decreasing doc-frequency
    dictionary.save_as_text("codeword_dictionary.txt", sort_by_word=False)
    print "##################### dictionary is cleaned, shrinked and saved"

def build_bow_save():
    dictionary = corpora.Dictionary.load('codeword_dictionary.dict')

    print "\n=========== begin BOW, ......"
    bow_stream = (dictionary.doc2bow(words) for words in words_stream())

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

    run_lda(30,parallel=False)



