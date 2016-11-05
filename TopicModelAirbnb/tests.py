
from pymongo import MongoClient
import csv
from review import Review,ReviewsDAL
from word_coder import WordCoder

def test_read_csv():
    with open("reviews.csv",'rt') as inf:
        inf.readline()# read out the header

        reader = csv.reader(inf)
        for index,(_,id,_,_,_,comment) in enumerate(reader):
            print "*************** [{}] ***************\n{}".format(index+1,comment)

            if index >= 20:
                return

def test_load_review_words():
    client = MongoClient()
    collection = client['airbnb']['reviews']
    cursor = collection.find({})

    for index in xrange(10):
        d = next(cursor)
        review = Review.from_dict(d)
        print "*************** {} ***************".format(index+1)
        print "raw: {}".format(review.sent.raw)
        print "words: {}".format(review.sent.words)

    client.close()

def test_code_words():
    dal = ReviewsDAL()
    wordcoder = WordCoder()

    for index,r in enumerate(dal.sampling(5)):
        print "****************** [{}]".format(index+1)
        print r.sent.raw
        print "\n"
        print wordcoder.code(r.sent.words)

if __name__ == "__main__":
    # test_read_csv()
    # test_load_review_words()
    test_code_words()