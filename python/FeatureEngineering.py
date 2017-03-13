import pandas as pd
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import Stemmer
from Feature_TFIDF import Feature_TFIDF
import time
from HomeDepotCSVReader import HomeDepotReader

class HomeDepotFeature():
    def __init__(self):
        # self.stemmer = PorterStemmer()
        # self.stemmer = SnowballStemmer('english')
        self.stemmer = Stemmer.Stemmer('english')


    def getFeature(self, df):
        ## Please feel free to add feature into this method.
        ## For testing, you may want to comment out some feature generation to save time
        ## as some takes a long time to run.

        # Create Brand Column
        # df = self.__createBrandColumn(df)

        #TODO: Chun Siong Working on Spell correction

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

    def __createBrandColumn(self, df):
        brand_df = df[df.name == "MFG Brand Name"][['product_uid', 'value']]
        # print(brand_df.info())
        brand_df.rename(columns={'value': 'brand'}, inplace=True)
        # print(brand_df.info())
        all_df = pd.merge(df, brand_df, how='left', on='product_uid')
        all_df['brand'].astype(str)
        # print(all_df.info())
        return all_df

    def __spell_correction(self, s):
        raise NotImplementedError

    def __stemming(self, s):
        return " ".join([self.stemmer.stemWord(word) for word in s.lower().split()])

    def __create_TFIDF(self, df, columnName):
        print("Create Feature_TFIDF: ", columnName)
        no_dul_df = df.drop_duplicates(['product_uid', columnName])

        return Feature_TFIDF().getCosineSimilarity(queries=df.search_term,
                                                   targets=df[columnName],
                                                   documents=no_dul_df[columnName])

    # def __create_DocumentLength(self, df, columnName):

if __name__ == "__main__":
    train_filename = '../../data/train.csv'
    test_filename = '../../data/test_small.csv'
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

