import pandas as pd
import numpy as np
from HomeDepotCSVReader import HomeDepotReader
from FeatureEngineering import HomeDepotFeature
from HomeDepotCSVWriter import HomeDepotCSVWriter
from XGBoostRanker import XGBoostRanker
from OrdinalRegressionRanker import OrdinalRegressionRanker
import LogisticRegressionRanker
from DataPreprocessing import DataPreprocessing
import Feature_Doc2Vec
import FacMachineRanker
from Utilities import Utility

def getFeature(train_query_df, product_df, attribute_df, test_query_df, features):
    print("####  Running: RunMe.getFeature() ####")
    return HomeDepotFeature().getFeature(train_query_df, product_df, attribute_df, test_query_df,features=features)

def dumpFeature2CSV(dataframe, fileName):
    print("####  Running: RunMe.dumpFeature2CSV() ####")
    HomeDepotCSVWriter().dumpCSV(dataframe, fileName)

def dumpFeature2RanklibCSV(dataframe, fileName):
    print("####  Running: RunMe.dumpFeature2RanklibCSV() ####")
    HomeDepotCSVWriter().write2RankLibCSV(dataframe, fileName)

def runXGBoostRanker(train_df, test_df):
    print("####  Running: RunMe.runXGBoostRanker() ####")
    xgb = XGBoostRanker()
    xgb.train(train_df)

def runOrdinalRegressionRankerLAD(train_df, test_df):
    print("####  Running: OrdinalRegression LAD ####")
    # dp=DataPreprocessing()
    # trainDF,validateDF=dp.generateValidationSet(train_df)
    orRanker = OrdinalRegressionRanker('lad')
    orRanker.train(train_df, None)
    print("####  Completed: OrdinalRegression LAD ####")

def runOrdinalRegressionRankerOrdRidgeGridSearch(train_df, test_df):
    print("####  Running GridSearch: OrdinalRegression ordridge ####")
    # dp=DataPreprocessing()
    # trainDF,validateDF=dp.generateValidationSet(train_df)
    orRanker = OrdinalRegressionRanker('ordridge')
    orRanker.gridSearch(train_df, None)
    print("####  Completed GridSearch: OrdinalRegression ordridge ####")



def runOrdinalRegressionRankerOrdRidge(train_df, test_df):
    print("####  Running: OrdinalRegression ordridge training ####")
    # dp=DataPreprocessing()
    # trainDF,validateDF=dp.generateValidationSet(train_df)
    orRanker = OrdinalRegressionRanker('ordridge')
    orRanker.train(train_df, None)
    print("####  Completed: OrdinalRegression ordridge training ####")
    return orRanker

def runFacMachineRanker(train_df, test_df):
    print("####  Running: Factorisation Machine ####")
    fmRanker = FacMachineRanker.FacMachineRanker()
    fmRanker.train(train_df, None)
    print("####  Completed: Fac Machine ####")


def runOrdinalRegressionRankerLogit(train_df, test_df):
    print("####  Running: OrdinalRegression LOGIT ####")
    # dp=DataPreprocessing()
    # trainDF,validateDF=dp.generateValidationSet(train_df)
    orRanker = OrdinalRegressionRanker('logit')
    orRanker.train(train_df, None)
    print("####  Completed: OrdinalRegression LOGIT ####")

def runOrdinalRegressionRankerLogat(train_df, test_df):
    print("####  Running: OrdinalRegression LOGAT ####")
    # dp=DataPreprocessing()
    # trainDF,validateDF=dp.generateValidationSet(train_df)
    orRanker = OrdinalRegressionRanker('logat')
    orRanker.train(train_df, None)
    print("####  Completed: OrdinalRegression LOGAT ####")


def runLogisticRegressionRanker(train_df, test_df):
    print("####  Running: Logistic Regression ####")
    # dp=DataPreprocessing()
    # trainDF,validateDF=dp.generateValidationSet(train_df)
    lrRanker = LogisticRegressionRanker.LogisticRegressionRanker()
    lrRanker.train(train_df, None)
    print("####  Completed: Logistic Regression ####")
    # lrRanker.train(trainDF, validateDF)


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

    print("Starting Feature Engineering")
    # all_df = getFeature(train_query_df, product_df, attribute_df, test_query_df,
    #                     features="brand,spelling,nonascii,word2vec,bm25,bm25expandedquery,Word2VecQueryExpansion")
    all_df = getFeature(train_query_df, product_df, attribute_df, test_query_df,
                        features="brand,attribute,spelling,nonascii,stopwords,stemming,tfidf,tfidf_expandedquery,doc2vec,doc2vec_expandedquery,word2vec,bm25,doclength,bm25expandedquery,Word2VecQueryExpansion")

    # Write all feature to a CSV. Next time can just read from here
    writer = HomeDepotCSVWriter()
    writer.dumpCSV(all_df, "../data/features.csv")

    # Run personal models from this point onward
    # runOrdinalRegressionRanker(train_query_df, test_query_df)
    # runXGBoostRanker(train_query_df, test_query_df)
    # doc = Feature_Doc2Vec()
    # doc.tr