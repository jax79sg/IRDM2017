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


    def __getVectorForDocument(self,document):
        """
        This class can only process vectors, so need to convert first.
        If words of document doesn't exists, it will be excluded
        :param document: document, can also be a query. Expecting a list of words.
        :return:
        """

        document_vector = [self.dictionary.doc2bow(text) for text in [document]]
        return document_vector


    def __convertToCorpus(self,documents):
        #Preprocessing the text
        dp = DataPreprocessing()
        text = dp.getBagOfWords(documentDF=documents['content'], return_type='document_tokens')

        #Create a Gensim text corpus based on documents
        print("Creating a text dictionary")
        self.dictionary=Dictionary(line.lower().split() for line in documents['content'])
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
        print("Initialising BM25...this may take a while")
        if(~isinstance(documents,pd.DataFrame)):
            if(documents.shape[1]!=2):
                if (list(documents)[0] != 'product_id' or list(documents)[1] != 'content'):
                    raise InvalidDatasetException("Invalid Dataframe for BM25 init","Expecting A Pandas.Dataframe with columns 'product_id' and 'content' only")

        #Mapping of product_id to indices of BM25 documents
        print("Mapping product_id to corpus index")
        counter=0
        for productid in documents['product_id']:
            self.productDict[productid]=counter
            counter=counter+1


        #Convert documents into corpus
        print("Converting documents into corpus")
        self.__convertToCorpus(documents)

        #Initialise the BM25 computation
        print("Computing BM25 baselines")
        self.bm25 = BM25(self.corpus)
        self.avgIDF = sum(map(lambda k: float(self.bm25.idf[k]), self.bm25.idf.keys())) / len(self.bm25.idf.keys())

        print("self.bm25 corpus_size",self.bm25.corpus_size)
        print("self.bm25 avgdl", self.bm25.avgdl)
        print("self.bm25 f", self.bm25.f)
        print("self.bm25 df", self.bm25.df)
        print("self.bm25 idf", self.bm25.idf)
        print("self.bm25 avgIDF", self.avgIDF)
        print("self.bm25 corpus", self.bm25.corpus)


        print("========Initialising BM25...completed...")




    def score(self,query,documents):
        """
        BM25 will return BM25(query1, doc1), BM25(query1, doc2), BM25(query1, doc3), BM25(query2, doc1)...
        if documnent doesn't exists, it will be ignored.
        Changelog
        - 15/3 KS First commit
        :param queries: Expecting a list of words. ['One','Two','Three']
        :param documents: A list of 'product_id'
        :return: A list of scores in the order of the given documents
        """

        #Convert query into vector
        print("===== Scoring BM25 for query ",query, " against documents ",documents)
        queryVector = self.__getVectorForDocument(query)[0]
        print("queryVector:",queryVector)

        #Find out the indices of the documents in the document corpus
        bm25_scores=[]
        for productid in documents:
            documentindex=self.productDict[productid]
            print("documentindex found:", documentindex)
            # Run the scoring function for the query against each document
            bm25_scores.append(self.bm25.get_score(queryVector,documentindex,self.avgIDF))

        print("===== Scoring BM25 for query completed")
        return bm25_scores

if __name__ == "__main__":
    #Test documents
    a = [[10001, 'hello world'], [10002, 'memme too too'], [10003, 'goto\nhello there again world'],[10004,'I am working too']]
    df = pd.DataFrame(a,columns=['product_id','content'])
    #Initialise model
    bm25 = Feature_BM25(df)
    #Test Query
    q=['too','too','unknownword']
    print(bm25.score(q,[10001,10002,10004]))


