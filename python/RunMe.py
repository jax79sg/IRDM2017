import pandas as pd
from HomeDepotCSVReader import HomeDepotReader
from FeatureEngineering import HomeDepotFeature
from HomeDepotCSVWriter import HomeDepotCSVWriter
from XGBoostRanker import XGBoostRanker
from OrdinalRegressionRanker import OrdinalRegressionRanker
from DataPreprocessing import DataPreprocessing
import Feature_Doc2Vec

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

def runOrdinalRegressionRanker(train_df, test_df):
    print("####  Running: OrdinalRegression ####")
    dp=DataPreprocessing()
    trainDF,validateDF=dp.generateValidationSet(train_df)
    orRanker = OrdinalRegressionRanker('logAT')
    orRanker.train(trainDF, validateDF)

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
    all_df = getFeature(train_query_df, product_df, attribute_df, test_query_df,features="brand,spelling,nonascii,Word2VecQueryExpansion")

    # Run personal models from this point onward
    # runOrdinalRegressionRanker(train_query_df, test_query_df)
    # runXGBoostRanker(train_query_df, test_query_df)
    # doc = Feature_Doc2Vec()
    # doc.tr