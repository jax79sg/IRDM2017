import pandas as pd
# from nltk.stem.porter import *
# from nltk.stem.snowball import SnowballStemmer
import Stemmer
from Feature_TFIDF import Feature_TFIDF
from Feature_Doc2Vec import Feature_Doc2Vec
from Feature_BM25 import Feature_BM25
import time
from HomeDepotCSVReader import HomeDepotReader
from DataPreprocessing import DataPreprocessing
import numpy as np
import Feature_Spelling
import re
from nltk.corpus import stopwords
import nltk
from AutomaticQueryExpansion import Word2VecQueryExpansion

class HomeDepotFeature():
    def __init__(self):
        # self.stemmer = PorterStemmer()
        # self.stemmer = SnowballStemmer('english')
        self.stemmer = Stemmer.Stemmer('english')

    def getFeature(self, train_query_df, product_df, attribute_df, test_query_df, features="brand,spelling,nonascii,stopwords,stemming,tfidf,doc2vec,bm25,doclength,Word2VecQueryExpansion"):
        ## Please feel free to add feature into this method.
        ## For testing, you may want to comment out some feature generation to save time
        ## as some takes a long time to run.


        if features.find("brand") != -1:
            # Create Brand Column
            product_df = self.__createBrandColumn(product_df, attribute_df)

        if features.find("spelling") != -1:
            # Perform spell correction on search_term
            print("Performing spell correction on search term")
            train_query_df['search_term'] = train_query_df['search_term'].map(lambda x: self.__spell_correction(x))

        if features.find("nonascii") != -1:
            # Remove non-ascii characters
            print("Performing non-ascii removal")
            start_time = time.time()
            train_query_df['search_term'] = train_query_df['search_term'].map(lambda x: self.__nonascii_clean((x)))
            print("Non-ascii clean on search_term took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['product_title'] = product_df['product_title'].map(lambda x: self.__nonascii_clean(str(x)))
            print("Non-ascii clean on product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        # Run this to download the download the stopword list if you hit error
        # nltk.download()

        if features.find("stopwords") != -1:
            # Stopwords removal
            print("Performing stopwords removal")
            start_time = time.time()
            train_query_df['search_term'] = train_query_df['search_term'].map(lambda x: self.__stopword_removal((x)))
            print("stopwords removal on search_term took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['product_title'] = product_df['product_title'].map(lambda x: self.__stopword_removal(str(x)))
            print("stopwords removal on product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        if features.find("stemming") != -1:
            # # Stemming
            print("Performing Stemming")
            start_time = time.time()
            train_query_df['search_term'] = train_query_df['search_term'].map(lambda x: self.__stemming((x)))
            print("Stemming search_term took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['product_title'] = product_df['product_title'].map(lambda x: self.__stemming(str(x)))
            print("Stemming product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['product_brand'] = product_df['product_brand'].map(lambda x: self.__stemming(str(x)))
            print("Stemming product_brand took: %s minutes" % round(((time.time() - start_time) / 60), 2))
            product_df['product_description'] = product_df['product_description'].map(lambda x: self.__stemming(str(x)))
            print("Stemming product_description took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        if features.find("tfidf") != -1:
            # TF-IDF
            print("Performing TF-IDF")
            tfidf = Feature_TFIDF()
            train_query_df['tfidf_product_title'] = tfidf.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_title')
            train_query_df['tfidf_product_brand'] = tfidf.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_brand')
            train_query_df['tfidf_product_description'] = tfidf.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_description')

        if features.find("doc2vec") != -1:
            # Doc2Vec
            print("Performing Doc2Vec")
            doc2vec = Feature_Doc2Vec()
            train_query_df['doc2vec_product_title'] = doc2vec.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_title')
            doc2vec = Feature_Doc2Vec()
            train_query_df['doc2vec_product_brand'] = doc2vec.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_brand')
            doc2vec = Feature_Doc2Vec()
            train_query_df['doc2vec_product_description'] = doc2vec.getCosineSimilarity(train_query_df, 'search_term', product_df,
                                                                              'product_description')

        if features.find("bm25") != -1:
            # BM25
            print("===========Performing BM25 computation....this may take a while")
            print("Merging product_title and description")
            print(list(product_df))
            product_df['content']=product_df['product_title'].map(str) +" "+ \
                                  product_df['product_description'].map(str) + " " + \
                                  product_df['product_brand'].map(str)
            product_df.head(1)
            print("Compute BM25")
            bm25 = Feature_BM25(product_df)
            print("Remove merged column")
            product_df=product_df.drop('content', axis=1)
            #For every training query-document pair, generate bm25
            print("Generate bm25 column")
            train_query_df=bm25.computeBM25Column(trainset=train_query_df,colName='bm25')
            print("train_query_df:",list(train_query_df))
            print("train_query_df head:",train_query_df.head(1))
            print("Saving to csv")
            train_query_df.to_csv('../data.prune/train_query_with_bm25.csv')
            print("===========Completed BM25 computation")


        if features.find("Word2VecQueryExpansion") != -1:
            # BM25
            print("===========Performing Word2VecQueryExpansion computation....this may take a super long time")
            # print("Merging product_title and description")
            # print(list(product_df))
            # product_df['content']=product_df['product_title'].map(str) +" "+ \
            #                       product_df['product_description'].map(str) + " " + \
            #                       product_df['product_brand'].map(str)
            # product_df.head(1)
            print("Compute Word2VecQueryExpansion")
            w2cExpand = Word2VecQueryExpansion()
            # print("Remove merged column")
            # product_df=product_df.drop('content', axis=1)
            #For every training query-document pair, generate bm25
            print("Generate Word2VecQueryExpansion column")
            train_query_df=w2cExpand.computeExpandedQueryColumn(trainset=train_query_df,colName='Word2VecQueryExpansion')
            print("train_query_df:",list(train_query_df))
            print("train_query_df head:",train_query_df.head(1))
            print("Saving to csv")
            train_query_df.to_csv('../data.prune/train_query_with_Word2VecQueryExpansion.csv')
            print("===========Completed Word2VecQueryExpansion computation")

        if features.find("doclength") != -1:
            # Document Length
            print("Performing Document Length")
            product_df['len_product_title'] = product_df['product_title'].map(lambda x: len(homedepotTokeniser(x)))
            train_query_df = pd.merge(train_query_df, product_df[['product_uid', 'len_product_title']], how='left',
                                      on='product_uid')
            product_df['len_product_description'] = product_df['product_description'].map(lambda x: len(homedepotTokeniser(x)))
            train_query_df = pd.merge(train_query_df, product_df[['product_uid', 'len_product_description']], how='left',
                                      on='product_uid')
            product_df['len_brand'] = product_df['product_brand'].map(lambda x: len(homedepotTokeniser(x)))
            train_query_df = pd.merge(train_query_df, product_df[['product_uid', 'len_brand']], how='left',
                                      on='product_uid')
            train_query_df['len_search_term'] = train_query_df['search_term'].map(lambda x: len(homedepotTokeniser(x)))


        print(train_query_df.info())

        return train_query_df

    def getFeature_old(self, df):
        ## Please feel free to add feature into this method.
        ## For testing, you may want to comment out some feature generation to save time
        ## as some takes a long time to run.

        # Create Brand Column
        # df = self.__createBrandColumn(df)

        #TODO: Chun Siong Working on Spell correction

        # Remove non-ascii characters
        print("Performing non-ascii removal")
        start_time = time.time()
        df['search_term'] = df['search_term'].map(lambda x: self.__nonascii_clean((x)))
        print("Non-ascii clean on search_term took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        df['product_title'] = df['product_title'].map(lambda x: self.__nonascii_clean(str(x)))
        print("Non-ascii clean on product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))


        # # Stemming
        print("Performing Stemming")
        start_time = time.time()
        df['search_term'] = df['search_term'].map(lambda x: self.__stemming((x)))
        print("Stemming search_term took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        df['product_title'] = df['product_title'].map(lambda x: self.__stemming(str(x)))
        print("Stemming product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        df['product_brand'] = df['product_brand'].map(lambda x: self.__stemming(str(x)))
        print("Stemming brand took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        # TF-IDF
        print("Performing TF-IDF")
        tfidf = self.__create_TFIDF(df, "product_title")
        df['tfidf_product_title'] = tfidf
        tfidf = self.__create_TFIDF(df, "product_description")
        df['tfidf_product_description'] = tfidf
        tfidf = self.__create_TFIDF(df, "value")
        df['tfidf_attributes_value'] = tfidf
        tfidf = self.__create_TFIDF(df, "brand")
        df['tfidf_brand'] = tfidf

        # Document Length
        print("Performing Document Length")
        df['len_product_title'] = df['product_title'].map(lambda x: len(x.split()))
        df['len_product_description'] = df['product_description'].map(lambda x: len(x.split()))
        # df['len_brand'] = df['brand'].map(lambda x: len(str(x).split()))
        df['len_search_term'] = df['search_term'].map(lambda x: len(str(x).split()))

        print(df.info())

        return df

    def __createBrandColumn(self, product_df, attribute_df):
        brand_df = attribute_df[attribute_df.name == "MFG Brand Name"][['product_uid', 'value']]
        brand_df.rename(columns={'value': 'product_brand'}, inplace=True)
        product_df = pd.merge(product_df, brand_df, how='left', on='product_uid')
        # print("No. of product without brand: ", product_df.product_brand.isnull().sum())
        product_df.product_brand.replace(np.NaN, 'unknown_brand_value', inplace=True)
        return product_df

    def __spell_correction(self, s):
        return " ".join([Feature_Spelling.spell_dict[word] if word in Feature_Spelling.spell_dict else word
                         for word in homedepotTokeniser(s)])

    def __stemming(self, s):
        # return " ".join([self.stemmer.stemWord(word) for word in s.lower().split()])
        return " ".join([self.stemmer.stemWord(word) for word in homedepotTokeniser(s)])

    def __nonascii_clean(self,s):
        return "".join(letter for letter in s if ord(letter) < 128)

    def __stopword_removal(self, s):
        return " ".join([word for word in homedepotTokeniser(s) if not word in stopwords.words('english')])

tokeniser = re.compile("(?:[A-Za-z]{1,2}\.)+|[\w\']+|\?\!")
def homedepotTokeniser(string):
    return tokeniser.findall(string)

if __name__ == "__main__":
    train_filename = '../../data/train.csv'
    test_filename = '../../data/test.csv'
    attribute_filename = '../../data/attributes.csv'
    description_filename = '../../data/product_descriptions.csv'

    reader = HomeDepotReader()
    train_df, test_df = reader.getMergedDataFrame(train_filename, test_filename, attribute_filename, description_filename)

    all_df = pd.concat((train_df, test_df), axis=0, ignore_index=True)

    print("Starting Feature Engineering")
    feat = HomeDepotFeature()
    all_df = feat.getFeature(train_df,features="brand,spelling,nonascii,stopwords,stemming,tfidf,doc2vec,bm25,doclength")

    train_df = all_df.iloc[:train_df.shape[0]]
    test_df = all_df.iloc[train_df.shape[0]:]

