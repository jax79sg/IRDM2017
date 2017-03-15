import pandas as pd
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import Stemmer
from Feature_TFIDF import Feature_TFIDF
from Feature_BM25 import Feature_BM25
import time
from HomeDepotCSVReader import HomeDepotReader
from DataPreprocessing import DataPreprocessing

class HomeDepotFeature():
    def __init__(self):
        # self.stemmer = PorterStemmer()
        # self.stemmer = SnowballStemmer('english')
        self.stemmer = Stemmer.Stemmer('english')

    def getFeature(self, train_query_df, product_df, attribute_df, test_query_df):
        ## Please feel free to add feature into this method.
        ## For testing, you may want to comment out some feature generation to save time
        ## as some takes a long time to run.

        # Create Brand Column
        product_df = self.__createBrandColumn(product_df, attribute_df)

        # TODO: Chun Siong Working on Spell correction

        # Remove non-ascii characters
        print("Performing non-ascii removal")
        start_time = time.time()
        train_query_df['search_term'] = train_query_df['search_term'].map(lambda x: self.__nonascii_clean((x)))
        print("Non-ascii clean on search_term took: %s minutes" % round(((time.time() - start_time) / 60), 2))
        product_df['product_title'] = product_df['product_title'].map(lambda x: self.__nonascii_clean(str(x)))
        print("Non-ascii clean on product_title took: %s minutes" % round(((time.time() - start_time) / 60), 2))

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

        # TF-IDF
        # print("Performing TF-IDF")
        # tfidf = self.__create_TFIDF(train_query_df, product_df, "product_title")
        # train_query_df['tfidf_product_title'] = tfidf
        # tfidf = self.__create_TFIDF(train_query_df, product_df, "product_description")
        # train_query_df['tfidf_product_description'] = tfidf
        # tfidf = self.__create_TFIDF(train_query_df, product_df, "value")
        # train_query_df['tfidf_attributes_value'] = tfidf
        # tfidf = self.__create_TFIDF(train_query_df, product_df, "brand")
        # train_query_df['tfidf_brand'] = tfidf
        # tfidf = Feature_TFIDF()
        # train_query_df['tfidf_product_title'] = tfidf.getCosineSimilarity(train_query_df, 'search_term', product_df, 'product_title')


        # BM25
        print("===========Performing BM25 computation....this may take a while")
        print("Merging product_title and description")
        print(list(product_df))
        product_df['content']=product_df['product_title'].map(str) +" "+ product_df['product_description']
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

        # Document Length
        print("Performing Document Length")
        product_df['len_product_title'] = product_df['product_title'].map(lambda x: len(x.split()))
        train_query_df = pd.merge(train_query_df, product_df[['product_uid', 'len_product_title']], how='left',
                                  on='product_uid')
        product_df['len_product_description'] = product_df['product_description'].map(lambda x: len(x.split()))
        train_query_df = pd.merge(train_query_df, product_df[['product_uid', 'len_product_description']], how='left',
                                  on='product_uid')
        product_df['len_brand'] = product_df['product_brand'].map(lambda x: len(str(x).split()))
        train_query_df = pd.merge(train_query_df, product_df[['product_uid', 'len_brand']], how='left',
                                  on='product_uid')
        train_query_df['len_search_term'] = train_query_df['search_term'].map(lambda x: len(str(x).split()))


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
        # df['brand'] = df['brand'].map(lambda x: self.__stemming(str(x)))
        # print("Stemming brand took: %s minutes" % round(((time.time() - start_time) / 60), 2))

        # TF-IDF
        print("Performing TF-IDF")
        tfidf = self.__create_TFIDF(df, "product_title")
        df['tfidf_product_title'] = tfidf
        tfidf = self.__create_TFIDF(df, "product_description")
        df['tfidf_product_description'] = tfidf
        # tfidf = self.__create_TFIDF(df, "value")
        # df['tfidf_attributes_value'] = tfidf
        # tfidf = self.__create_TFIDF(df, "brand")
        # df['tfidf_brand'] = tfidf

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
        return product_df

    def __spell_correction(self, s):
        raise NotImplementedError

    def __stemming(self, s):
        return " ".join([self.stemmer.stemWord(word) for word in s.lower().split()])

    def __nonascii_clean(self,s):
        return "".join(letter for letter in s if ord(letter) < 128)

    # def __create_TFIDF(self, df, columnName):
    #     tfidf = Feature_TFIDF()
    #     tfidf.getCosineSimilarity_2(source_df, source_columnName, target_df, target_columnName)
    #
    # def __create_TFIDF_old(self, df, columnName):
    #     print("Create Feature_TFIDF: ", columnName)
    #     no_dul_df = df.drop_duplicates(['product_uid', columnName])
    #
    #     return Feature_TFIDF().getCosineSimilarity(queries=df.search_term,
    #                                                targets=df[columnName],
    #                                                documents=no_dul_df[columnName])

    # def __create_DocumentLength(self, df, columnName):

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
    all_df = feat.getFeature(train_df)

    train_df = all_df.iloc[:train_df.shape[0]]
    test_df = all_df.iloc[train_df.shape[0]:]

