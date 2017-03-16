import pandas as pd
import numpy as np
import DataPreprocessing as DataPreprocessing

class HomeDepotReader:

    def getBasicDataFrame(self, filename, header=0):
        '''
        Basic CSV reader
        :param filename: Filename
        :param header: Index of header
        :return: A dataframe
        '''
        return pd.read_csv(filename, delimiter=',', low_memory=False, header=header)

    def getMergedDataFrame(self, train_filename, test_filename, attribute_filename, description_filename, header=0):
        '''
        Read in all CSV and merge attribute and description into train_df and test_df

        :param train_filename:
        :param test_filename:
        :param attribute_filename:
        :param description_filename:
        :param header: Index of header
        :return: [train_df, test_df]
        '''

        train_df = pd.read_csv(train_filename, delimiter=',', low_memory=False, header=header, encoding="ISO-8859-1")
        test_df = pd.read_csv(test_filename, delimiter=',', low_memory=False, header=header, encoding="ISO-8859-1")
        # attribute_df = pd.read_csv(attribute_filename, delimiter=',', low_memory=False, header=header, encoding="ISO-8859-1")
        description_df = pd.read_csv(description_filename, delimiter=',', low_memory=False, header=header, encoding="ISO-8859-1")

        train_len = train_df.shape[0]

        all_df = pd.concat((train_df, test_df), axis=0, ignore_index=True)
        # all_df = pd.merge(all_df, attribute_df, how='left', on='product_uid')
        all_df = pd.merge(all_df, description_df, how='left', on='product_uid')

        # Convert to suitable type to save some memory
        all_df['id'] = all_df.id.astype(np.int32)
        all_df['product_uid'] = all_df.product_uid.astype(np.int32)
        all_df['relevance'] = all_df.relevance.astype(np.float16)

        train_df = all_df.iloc[:train_len]
        test_df = all_df.iloc[train_len:]

        # print("All Dataframe Information")
        # print("###############################")
        # print(all_df.info())

        # print(len(all_df))
        # print(len(all_df.id.unique()))

        # print("\nTrain Dataframe Information")
        # print("###############################")
        # print(train_df.info())
        #
        # print("\nTest Dataframe Information")
        # print("###############################")
        # print(test_df.info())

        return [train_df, test_df]

    def getQueryProductAttributeDataFrame(self, train_filename, test_filename, attribute_filename, description_filename, header=0):
        '''
        Takes in HomeDepot CSV and process into the following dataframe:
        1. train_query_df/test_query_df
            - id
            - product_uid
            - search_term
            - relevance
        2. product_df
            - product_uid
            - product_title
            - product_description

        3. attribute_df
            - product_uid
            - name
            - value

        :param train_filename:
        :param test_filename:
        :param attribute_filename:
        :param description_filename:
        :param header:
        :return: [train_query_df, product_df, attribute_df, test_query_df]
        '''
        train_query_df = pd.read_csv(train_filename, delimiter=',', low_memory=False, header=header, encoding="ISO-8859-1")
        test_query_df = pd.read_csv(test_filename, delimiter=',', low_memory=False, header=header, encoding="ISO-8859-1")
        attribute_df = pd.read_csv(attribute_filename, delimiter=',', low_memory=False, header=header, encoding="ISO-8859-1")
        description_df = pd.read_csv(description_filename, delimiter=',', low_memory=False, header=header, encoding="ISO-8859-1")

        all_df = pd.concat((train_query_df, test_query_df), axis=0, ignore_index=True)

        train_query_df = train_query_df.drop('product_title', axis=1)
        test_query_df = test_query_df.drop('product_title', axis=1)

        product_df = pd.DataFrame()
        product_df = all_df.drop_duplicates(['product_uid'])
        product_df = product_df.drop('relevance', axis=1)
        product_df = product_df.drop('id', axis=1)
        product_df = product_df.drop('search_term', axis=1)
        product_df = pd.merge(product_df, description_df, how='left', on='product_uid')

        dp= DataPreprocessing.DataPreprocessing()
        train_query_df=dp.transformLabels(trainDF=train_query_df,newColName='relevance_int')



        print("all: ", len(all_df.product_uid))
        print("all unique: ", len(all_df.product_uid.unique()))
        # print("description_df: ", len(description_df))
        # print("description_df unique: ", len(description_df.product_uid.unique()))
        print("product_df: ", len(product_df))
        print("product_df info", product_df.info())

        return [train_query_df, product_df, attribute_df, test_query_df]