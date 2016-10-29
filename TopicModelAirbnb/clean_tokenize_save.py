
import csv
from review import Review,ReviewsDAL

def review_stream():
    with open("reviews.csv",'rt') as inf:
        inf.readline()# read out the header

        reader = csv.reader(inf)
        for index,(_,id,_,_,_,comment) in enumerate(reader):
            try:
                comment = comment.encode('ascii','ignore')
                yield Review(id,comment)
            except UnicodeDecodeError:
                # ignore non-ASCII comments
                pass

def read_save_mongodb(buffersize=300):
    r_stream = review_stream()
    dal = ReviewsDAL()

    buffer = []
    for index,review in enumerate(r_stream):
        if index % buffersize == 0:
            dal.insert_many(buffer)
            del buffer[:] # clear
            print "{} reviews saved into mongodb".format(index)

        buffer.append(review)

    dal.insert_many(buffer)
    dal.close()

    print "----------- DONE -----------"
    print "totally {} reviews inserted into mongodb".format(index+1)

if __name__ == "__main__":
    read_save_mongodb()
