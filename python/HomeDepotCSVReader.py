import pandas as pd
import numpy as np



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



