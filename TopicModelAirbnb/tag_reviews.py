
import pandas as pd
from review import Review,ReviewsDAL
from gensim import corpora,models,matutils

if __name__ == "__main__":
    model = models.LdaModel.load("airbnb.lda_model")
    dictionary = corpora.Dictionary.load('codeword_dictionary.dict')

    topic_mapping = pd.read_csv('airbnb_topics.csv',index_col='id').iloc[:,0].to_dict()

    dal = ReviewsDAL()
    review_stream = dal.sampling(10)

    for index,review in enumerate( review_stream):
        print "******************** [{}]".format(index+1)
        print review.sent.raw

        bow = dictionary.doc2bow(review.sent.words)
        topic_distribution = model[bow]
        topic_distribution.sort(key=lambda t:t[1],reverse=True)

        for index,(topic_id,topic_percentage) in enumerate( topic_distribution ):
            print '[{}] <{}> {}:{:.2f}'.format(index+1,topic_id,topic_mapping[topic_id],topic_percentage)


    dal.close()
