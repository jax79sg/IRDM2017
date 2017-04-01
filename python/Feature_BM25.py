from gensim.corpora import Dictionary, MmCorpus
from gensim.summarization.bm25 import BM25
import numpy as np
import pandas as pd
from UserException import InvalidDatasetException
from DataPreprocessing import DataPreprocessing
from collections import defaultdict

class Feature_BM25():

    bm25=None
    corpus=None
    dictionary=None
    productDict = defaultdict(int)
    avgIDF=0

    #Lookup
    colProductId='product_id'
    colContent='content'

    def getBM25(self,df, searchTermColname):
        """
        Meant to be use in panda apply.
        See below
        :param df:
        :return:
        """
        productid = df['product_uid']
        search_term = df[searchTermColname].split()
        return self.score(search_term, productid)

    def computeBM25Column(self,trainset, destColName='bm25', searchTermColname='search_term'):
        """
        Compute a new bm25 column given a dataframe
        Changelog
        - 15/3 KS First commit
        - 1/4 KS Added support for other search terms
        :param trainset: Training Dataframe, should contain  product_uid and search_term columns
        :param colName: name of the new column
        :return:
        """
        if(~isinstance(trainset,pd.DataFrame)):
            if(trainset.shape[1]<2):
                if ('product_uid' not in list(trainset) or searchTermColname not in list(trainset)):
                    raise InvalidDatasetException("Invalid Dataframe for BM25 compute","Expecting A Pandas.Dataframe with columns 'product_uid' and ",searchTermColname)

        #Apply scoring function across each row of dataframe
        trainset[destColName]=trainset.apply(self.getBM25,args=(searchTermColname,),axis=1)
        return trainset


    def __getVectorForDocument(self,document):
        """
        Convert the document into list of vectors .
        Changelog
        - 15/3 KS First commit
        :param document: document, can also be a query. Expecting a list of words.
        :return:
        """

        document_vector = [self.dictionary.doc2bow(text) for text in [document]]
        return document_vector


    def __convertToCorpus(self,documents):
        """
        Steps to make the documents compatible to gensim
        Changelog
        - 15/3 KS First commit
        :param documents:
        :return:
        """
        #Preprocessing the text
        dp = DataPreprocessing()
        text = dp.getBagOfWords(documentDF=documents, return_type='document_tokens')

        #Create a Gensim text corpus based on documents
        print("Creating a text dictionary")
        self.dictionary=Dictionary(line.lower().split() for line in documents)
        print(self.dictionary)
        print("Saving text dictionary to file")
        self.dictionary.save('../data.prune/producttext.dict')

        #Create a Gensim document corpus based on text corpus and each document
        print("Creating a Gensim document corpus")
        self.corpus=[self.dictionary.doc2bow(line) for line in text]

        print("Saving corpus to file")
        MmCorpus.serialize('../data.prune/productcorpus.mm', self.corpus)
        self.corpus = MmCorpus('../data.prune/productcorpus.mm')
        print(self.corpus)

    def __init__(self,documents):
        """
        Changelog
        - 15/3 KS First Commit
        Initialise the BM25 corpus with words of all documents
        :param documents: A Dataframe 2 columns. One of 'product_id', other of 'content'
        """
        print("Initialising BM25...")
        if(~isinstance(documents,pd.DataFrame)):
            if(documents.shape[1]<2):
                if ('product_uid' not in list(documents) or 'content' not in list(documents)):
                    raise InvalidDatasetException("Invalid Dataframe for BM25 init","Expecting A Pandas.Dataframe with columns 'product_uid' and 'content' ")

        #Mapping of product_id to indices of BM25 documents
        print("Mapping product_id to corpus index")
        counter=0
        for productid in documents['product_uid']:
            self.productDict[productid]=counter
            counter=counter+1


        #Convert documents into corpus
        print("Converting documents into corpus")
        self.__convertToCorpus(documents['content'])

        #Initialise the BM25 computation
        print("Computing BM25 baselines")
        self.bm25 = BM25(self.corpus)
        self.avgIDF = sum(map(lambda k: float(self.bm25.idf[k]), self.bm25.idf.keys())) / len(self.bm25.idf.keys())

        print("self.bm25 corpus_size",self.bm25.corpus_size)
        print("self.bm25 avgdl", self.bm25.avgdl)
        # print("self.bm25 f", self.bm25.f)
        # print("self.bm25 df", self.bm25.df)
        # print("self.bm25 idf", self.bm25.idf)
        print("self.bm25 avgIDF", self.avgIDF)
        print("self.bm25 corpus", self.bm25.corpus)


        print("========Initialising BM25...completed...")

    def score(self,query,document):
        """
        BM25 will return BM25(query1, doc1).
        if documnent doesn't exists, it will be ignored.
        Changelog
        - 15/3 KS First commit
        :param query: Expecting a list of words. ['One','Two','Three']
        :param document: A  'product_id'
        :return:
        """
        #Convert query into vector
        # print("===== Single scoring BM25 for query ",query, " against documents ",document)
        queryVector = self.__getVectorForDocument(query)[0]
        # print("queryVector:",queryVector)

        # Find out the index of the document in the document corpus
        documentindex = self.productDict[document]

        score=self.bm25.get_score(queryVector, documentindex, self.avgIDF)
        # print("score:",score)
        # print("===== Scoring BM25 for query completed")
        return score

    def scores(self,query,documents):
        """
        BM25 will return BM25(query1, doc1), BM25(query1, doc2), BM25(query1, doc3), BM25(query2, doc1)...
        if documnent doesn't exists, it will be ignored.
        Changelog
        - 15/3 KS First commit
        :param queries: Expecting a list of words. ['One','Two','Three']
        :param documents: A list of 'product_id'
        :return: A list of scores in the order of the given documents
        """



        # #Convert query into vector
        print("===== Multiple scoring BM25 for query ",query, " against documents ",documents)
        # queryVector = self.__getVectorForDocument(query)[0]
        # print("queryVector:",queryVector)

        #Find out the indices of the documents in the document corpus
        bm25_scores=[]
        for productid in documents:
            # documentindex=self.productDict[productid]
            # print("documentindex found:", documentindex)
            # Run the scoring function for the query against each document
            # bm25_scores.append(self.bm25.get_score(queryVector,documentindex,self.avgIDF))
            print("Product ID:",productid)
            bm25_scores.append(self.score(query,productid))

        print("===== Scoring BM25 for query completed")
        print("bm25_scores:",bm25_scores)
        return bm25_scores

if __name__ == "__main__":
    #Test documents
    a = [[10001, 'hello world'], [10002, 'memme too too'], [10003, 'goto\nhello there again world'],[10004,'I am working too']]
    df = pd.DataFrame(a,columns=['product_uid','content'])
    #Initialise model
    bm25 = Feature_BM25(df)
    #Test Query
    q=['too','too','unknownword']
    print(bm25.scores(q,[10001,10002,10004]))


