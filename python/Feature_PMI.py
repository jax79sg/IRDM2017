
import Utilities
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from math import log2
from HomeDepotCSVReader import HomeDepotReader
import pandas as pd
from UserException import InvalidDatasetException

class Feature_PMI():
    bigrams=None
    unigrams=None
    corpussize=None

    def __init__(self,text):
        """
        Features for PMI.
        Will init corpus.
        Changelog: 
        First commit - KS
        :param text: The corpus
        """
        self._generateBigrams(text)
        self._generateUnigrams(text)
        self.corpussize=len(nltk.word_tokenize(text))
        print("Feature_PMI: Corpus size:",self.corpussize)

    def _generateNgrams(self,text,n=2):
        """
        Compute an ngram, given the test and N.
        Changelog: 
        First commit - KS        
        :param text: The corpus
        :param n: The number, default bigram.
        :return: 
        """
        token = nltk.word_tokenize(text)
        computedNgrams=ngrams(token,n)
        return Counter(computedNgrams)

    def _generateBigrams(self,text):
        """
        Generate and store the bigrams
        Changelog: 
        First commit - KS        
        :param text: corpus
        :return: 
        """
        self.bigrams=self._generateNgrams(text,2)

    def _generateUnigrams(self,text):
        """
        Generate and store the unigrams
        Changelog: 
        First commit - KS        
        :param text: corpus
        :return: 
        """
        self.unigrams=self._generateNgrams(text,1)

    def _getCountForBigram(self,word1,word2):
        """
        Return the count of occurances for bigram
        Changelog: 
        First commit - KS        
        :param word1: 
        :param word2: 
        :return: 
        """
        return self.bigrams[(word1,word2)]

    def _getCountForUnigram(self,word1):
        """
        Return the count of occurances for bigram
        Changelog: 
        First commit - KS        
        :param word1: 
        :return: 
        """
        count=self.unigrams[(word1)]
        if count==0:
            count=0.001
        return count


    def computePMI(self,word1,word2):
        """
        Compute the PMI value of a bigram
        Changelog: 
        First commit - KS        
        :param word1: 
        :param word2: 
        :return: 
        """
        pmi=0
        if(word1 is not None and word2 is not None):
            #PMI = P(word1|word2)/P(word1)P(word2)
            #    = P(word2|word1)/P(word1)P(word2)
            P_w1w2=self._getCountForBigram(word1,word2)/self.corpussize
            p_w1=self._getCountForUnigram(word1)/self.corpussize
            p_w2 = self._getCountForUnigram(word2)/self.corpussize
            try:
                pmi=log2(P_w1w2/(p_w1*p_w2))
            except ValueError:
                pmi=99999
                # print("P_w1w2:",P_w1w2," p_w1:",p_w1," p_w2:",p_w2)
        return pmi

    def _getPMI(self,df):
        """
        Internal method for dataframe apply
        Changelog: 
        First commit - KS        
        :param df: 
        :return: 
        """
        pmi=0
        search_term = df['search_term'].split()
        noofterms=len(search_term)
        startindex=0
        pmiAccumulate=0
        if(noofterms>1):
            for i in range(0,noofterms-1):
                pmi=self.computePMI(search_term[i],search_term[i+1])
                pmiAccumulate=pmiAccumulate+pmi
            pmiAccumulate=pmiAccumulate/noofterms
            pmi=pmiAccumulate
        return pmi

    def computePMIColumn(self,trainset, destColName='pmi', searchTermColname='search_term'):
        """
        Compute the PMT for entire recordset.
        Changelog: 
        First commit - KS
        :param trainset: 
        :param destColName: 
        :param searchTermColname: 
        :return: 
        """
        if(~isinstance(trainset,pd.DataFrame)):
            if(trainset.shape[1]<2):
                if (searchTermColname not in list(trainset)):
                    raise InvalidDatasetException("Invalid Dataframe for PMI compute","Expecting A Pandas.Dataframe with columns:",searchTermColname)

        #Apply scoring function across each row of dataframe
        trainset[destColName]=trainset.apply(self._getPMI,axis=1)
        return trainset



if __name__ == "__main__":
    train_filename = '../../data/train.csv'
    test_filename = '../../data/test.csv'
    attribute_filename = '../../data/attributes.csv'
    description_filename = '../../data/product_descriptions.csv'

    reader = HomeDepotReader()

    train_query_df, product_df, attribute_df, test_query_df = reader.getQueryProductAttributeDataFrame(train_filename,
                                                  test_filename,
                                                  attribute_filename,
                                                  description_filename)
    print("train_query_df:",list(train_query_df))
    print("product_df:", list(product_df))
    print("attribute_df:", list(attribute_df))
    print("test_query_df:", list(test_query_df))

    product_df['content'] = product_df['product_title'].map(str) + " " + \
                            product_df['product_description'].map(str)

    # print("Adding training query for that product id into the content")
    #
    # product_df = product_df.reset_index(drop=True)
    # counter = 0
    # for index, product in product_df.iterrows():
    #     # print("product:", product)
    #     productId = product['product_uid']
    #     # print("productId:",productId)
    #     df = train_query_df[train_query_df.product_uid == productId]
    #     # print("df:",df)
    #     searchterms = ""
    #     for index, row in df.iterrows():
    #         searchterm = row['search_term']
    #         searchterms = searchterms + " " + searchterm
    #
    #     newString = product_df.iloc[counter]['content'] + " " + searchterms
    #     product_df.set_value(counter, 'content', newString)
    #
    #     counter = counter + 1


    #Creating content
    text=product_df['content'].str.cat(sep=' ')
    pmiFeature = Feature_PMI(text)

    print("PMI 'kitchen','cabinet': ",pmiFeature.computePMI('kitchen','cabinet'))
    train_query_df = pmiFeature.computePMIColumn(trainset=train_query_df)
    print(list(train_query_df),"\n",train_query_df['pmi'])
    train_query_df.filter(items=['id','pmi']).to_csv('pmi_features.csv')

    # for index, row in product_df.iterrows():
