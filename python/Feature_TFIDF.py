from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import time


class Feature_TFIDF():
    # def __init__(self):
    #     print("init: Feature_TFIDF")
    #     # nltk.download('punkt')

    def getCosineSimilarity(self, queries, targets, documents):
        start_time = time.time()
        # print("Into getCosineSimilarity")
        tfidf = TfidfVectorizer(tokenizer=nltk.word_tokenize, smooth_idf=True, use_idf=True, sublinear_tf=True)
        print("TfidfVectorizer init took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        tfs = tfidf.fit_transform(documents)
        print("TfidfVectorizer fit took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        # print(tfs.toarray())

        score_queries = tfidf.transform(queries)
        print("TfidfVectorizer transform 1 took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        # feature_names = tfidf.get_feature_names()
        # for col in response.nonzero()[1]:
        #     print(feature_names[col], ' - ', response[0, col])


        # tfidf = TfidfVectorizer(tokenizer=nltk.word_tokenize, smooth_idf=True, use_idf=True, sublinear_tf=True)
        # tfs = tfidf.fit_transform(documents.values())
        # print(tfs.toarray())

        score_targets = tfidf.transform(targets)
        print("TfidfVectorizer transform 2 took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        # feature_names = tfidf.get_feature_names()
        # for col in response.nonzero()[1]:
        #     print(feature_names[col], ' - ', response[0, col])


        # print(cosine_similarity(score_queries, score_targets))

        # print([cosine_similarity(q, t) for q, t in zip(score_queries, score_targets)])
        result = np.array([cosine_similarity(q, t) for q, t in zip(score_queries, score_targets)])
        print("TfidfVectorizer Cosine similarity took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        # print (result.reshape(result.shape[0]))

        return result.reshape(result.shape[0])


        # cvec = CountVectorizer()
        # counts = cvec.fit_transform(self.documents.values())
        # print(counts.todense())
        #
        # transformer = TfidfTransformer(smooth_idf=True, use_idf=True, sublinear_tf=True)
        # counts = [[0,0,3,0,0],
        #           [1,1,2,0,0],
        #           [0,0,2,1,1],
        #           [0,0,0,1,1],
        #           [3,1,1,0,1]]
        # tfidf = transformer.fit_transform(counts)
        # print(tfidf.toarray())


if __name__ == "__main__":

    myDict = {}

    # myDict['s1'] = 'If it walks like a duck and quacks like a duck, it must be a duck.'
    # myDict['s2'] = 'Beijing Duck is mostly prized for the thin, crispy duck skin with authentic versions of the dish serving mostly the skin.'
    # myDict['s3'] = 'Bugs\' ascension to stardom also prompted the Warner animators to recast Daffy Duck as the rabbit\'s rival, intensely jealous and determined to steal back the spotlight while Bugs remained indifferent to the duck\'s jealousy, or used it to his advantage. This turned out to be the recipe for the success of the duo.'
    # myDict['s4'] = '6:25 PM 1/7/2007 blog entry: I found this great recipe for Rabbit Braised in Wine on cookingforengineers.com.'
    # myDict['s5'] = 'Last week Li has shown you how to make the Sechuan duck. Today we\'ll be making Chinese dumplings (Jiaozi), a popular dish that I had a chance to try last summer in Beijing. There are many recipies for Jiaozi.'

    myDict['s1'] = 'duck duck duck'
    myDict['s2'] = 'beijing dish duck duck'
    myDict['s3'] = 'duck duck rabbit recipe'
    myDict['s4'] = 'rabbit recipe'
    myDict['s5'] = 'beijing beijing beijing dish duck recipe'

    tfidf = Feature_TFIDF()
    # tfidf = Feature_TFIDF("beijing beijing beijing dish duck recipe", myDict)

    source = []
    source.append("beijing duck recipe")
    source.append("beijing duck recipe")
    source.append("beijing duck recipe")
    source.append("beijing duck recipe")
    source.append("beijing duck recipe")

    # tfidf.getCosineSimilarity(["beijing duck recipe"], ["duck duck duck"], myDict.values())
    # tfidf.getCosineSimilarity(["beijing duck recipe"], ["beijing dish duck duck"], myDict.values())
    # tfidf.getCosineSimilarity(["beijing duck recipe"], ["duck duck rabbit recipe"], myDict.values())
    # tfidf.getCosineSimilarity(["beijing duck recipe"], ["rabbit recipe"], myDict.values())
    # tfidf.getCosineSimilarity(["beijing duck recipe"], ["beijing beijing beijing dish duck recipe"], myDict.values())


    tfidf.getCosineSimilarity(source, myDict.values(), myDict.values())
