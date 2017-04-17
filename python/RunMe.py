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
    feature_df = HomeDepotFeature().getFeature(train_query_df, product_df, attribute_df, test_query_df,features=features)

    # Write all feature to a CSV. Next time can just read from here
    dumpFeature2CSV(feature_df, "../data/features_full_20170416.csv")

    return feature_df

def dumpFeature2CSV(dataframe, fileName):
    print("####  Running: RunMe.dumpFeature2CSV() ####")
    HomeDepotCSVWriter().dumpCSV(dataframe, fileName)

def dumpFeature2RanklibCSV(dataframe, fileName):
    print("####  Running: RunMe.dumpFeature2RanklibCSV() ####")
    HomeDepotCSVWriter().write2RankLibCSV(dataframe, fileName)

def runXGBoostRanker():
    print("####  Running: RunMe.runXGBoostRanker() ####")
    reader = HomeDepotReader()
    feature_df = reader.getBasicDataFrame("../data/features_doc2vec_sense2vec_20170416.csv")

    feature_train_df = feature_df[:74067]
    feature_test_df = feature_df[74067:]

    feature_test_df.pop('relevance')

    soln_filename = '../data/solution.csv'
    soln_df = pd.read_csv(soln_filename, delimiter=',', low_memory=False, encoding="ISO-8859-1")
    dp = DataPreprocessing()
    test_private_df = dp.getGoldTestSet(feature_test_df, soln_df,
                                        testsetoption='Private')
    test_public_df = dp.getGoldTestSet(feature_test_df, soln_df,
                                       testsetoption='Public')

    xgb = XGBoostRanker(feature_train_df)
    xgb.train_Regressor(feature_train_df)
    # xgb.gridSearch_Regressor(feature_train_df)

    # result_df = xgb.test_Model(test_public_df)
    result_df = xgb.test_Model(test_private_df)

    # # Compute NDCG Score
    # gold_df = pd.DataFrame()
    # gold_df['search_term'] = test_private_df['search_term']
    # gold_df['product_uid'] = test_private_df['product_uid']
    # gold_df['relevance_int'] = test_private_df['relevance']
    # ndcg = NDCG_Eval()
    # ndcg.computeAvgNDCG(gold_df, result_df)

    # # Dump the prediction to csv
    # result_df.pop('product_uid')
    # result_df.pop('search_term')
    # result_df.pop('relevance_int')
    # print(result_df.columns)
    # dumpFeature2CSV(result_df, "../data/xgboost_private_20170417.csv")

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
    # Mega combine all and generate feature for train and test all at one go.
    all_df = pd.concat((train_query_df, test_query_df))
    feature_df = getFeature(all_df, product_df, attribute_df, test_query_df, features="brand,attribute,spelling,nonascii,stopwords,colorExist,color_onehot,brandExist,wmdistance,stemming,word2vec,Word2VecQueryExpansion,tfidf,tfidf_expandedquery,doc2vec,doc2vec_expandedquery,bm25,bm25expandedquery,bm25description,bm25title,bm25brand,doclength")

    # Run personal models from this point onward
    # runOrdinalRegressionRanker(train_query_df, test_query_df)
    # runXGBoostRanker(train_query_df, test_query_df)
