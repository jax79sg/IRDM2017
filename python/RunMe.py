import pandas as pd
from HomeDepotCSVReader import HomeDepotReader
from FeatureEngineering import HomeDepotFeature
from HomeDepotCSVWriter import HomeDepotCSVWriter
from XGBoostRanker import XGBoostRanker

def getFeature(dataframe):
    print("####  Running: RunMe.getFeature() ####")
    return HomeDepotFeature().getFeature(dataframe)

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

if __name__ == "__main__":
    train_filename = '../../data/train.csv'
    test_filename = '../../data/test.csv'
    attribute_filename = '../../data/attributes.csv'
    description_filename = '../../data/product_descriptions.csv'

    reader = HomeDepotReader()
    train_df, test_df = reader.getMergedDataFrame(train_filename,
                                                  test_filename,
                                                  attribute_filename,
                                                  description_filename)

    all_df = pd.concat((train_df, test_df), axis=0, ignore_index=True)

    print("Starting Feature Engineering")
    all_df = getFeature(all_df)

    train_df = all_df.iloc[:train_df.shape[0]]
    test_df = all_df.iloc[train_df.shape[0]:]

    # After creating the feature, we can save to CSV so that we can test our model using it
    dumpFeature2CSV(train_df, "../../data/train_encoded.csv")
    # dumpFeature2CSV(test_df, "../../data/test_encoded.csv")

    # Dump train dataframe in Ranklib format so that we can run using Ranklib learning to rank algo
    # dumpFeature2RanklibCSV(train_df, "../../data/train_encoded.csv")

    # Run personal models from this point onward
    runXGBoostRanker(train_df, test_df)