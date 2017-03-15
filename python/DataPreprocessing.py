from UserException import InvalidDatasetException
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class DataPreprocessing():

    oldLabels=None
    newLabels = None
    mergedLabelDF = None

    def __init__(self):
        pass

    def generateValidationSet(self,trainset, test_size=0.2):
        """
        Changelog:
        - 14/3 KS First Commit
        Split the training set into train and validation sets.
        :param trainset:
        :param test_size: Default test size is 20% of original data
        :return:
        """
        print("Creating validation set")
        print("trainset shape:",trainset.shape)
        train,validation=train_test_split(trainset,test_size=test_size,random_state=42)
        print("splitted trainset shape:", train.shape)
        print("splitted validation shape:", validation.shape)
        return train,validation

    def __replaceLabel(self,x):
        """
        Changelog:
        - 14/3 KS First Commit
        Look up old labels and map to new labels
        :param x:
        :return:
        """
        indexRow=np.where(self.oldLabels == x)
        newLabel=self.newLabels[indexRow][0]
        return newLabel

    def transformLabels(self,trainDF=None, validationDF=None, trainLabelColumn='relevance', validationLabelColumn='relevance', newColName="relevance_int"):
        """
        Changelog:
        - 14/3 KS First Commit
        Existing labels are real numbers. This method will transform them into
         discrete numbers in the respective ordering starting from 0.
         Only run this method if your model requires labels to be discrete and ordered integers.

        :param trainDF: Required. Dataframe of training set
        :param validationDF: Optional. Datatframe of validation set
        :param newColName: Optional. This will be the name of the new column appended to both trainDF and testDF
        :param trainLabelColumn: Optional. Column of the label in the training set
        :param validationLabelColumn: Optional. Column of the label in the validation set

        :return:
        """

        # if(trainDF==None or validationDF==None):
        #     raise InvalidDatasetException("Invalid datasets","Both train and validation datasets must be provided ")
        print("===========Tranforming labels...\nshowing current values")


        if(trainDF is not None and validationDF is not None):
            print("trainDF:", list(trainDF))
            print("trainDF:", trainDF.head(1))
            print("validationDF:", list(validationDF))
            print("validationDF:", validationDF.head(1))

            #merge train and validation dataframe
            trainLabelDF=pd.DataFrame(trainDF[trainLabelColumn])
            print("Extracted trainLabelDF:",list(trainLabelDF),"\n",type(trainLabelDF),trainLabelDF.shape,trainLabelDF.head(1))
            trainLabelDF.columns=[trainLabelColumn]
            validationLabelDF = pd.DataFrame(validationDF[validationLabelColumn])
            print("Extracted validationLabelDF:", list(validationLabelDF), "\n", type(validationLabelDF),validationLabelDF.shape, validationLabelDF.head(1))
            validationLabelDF.columns = [trainLabelColumn]
            self.mergedLabelDF=trainLabelDF.append(validationLabelDF)
        elif(trainDF is not None and validationDF is None):
            print("trainDF:", list(trainDF))
            print("trainDF:", trainDF.head(1))

            self.mergedLabelDF=pd.DataFrame(trainDF[trainLabelColumn])
        else:
            raise InvalidDatasetException("Invalid dataset","The training set must exists")

        print("self.mergedLabelDF:", list(self.mergedLabelDF), "\n", type(self.mergedLabelDF), self.mergedLabelDF.shape, self.mergedLabelDF.head(1))

        #Get existing unique labels
        self.oldLabels = self.mergedLabelDF[trainLabelColumn].sort_values().unique()
        print("Old unique Labels:",self.oldLabels)

        #Generate new labels
        self.newLabels = np.arange(0, self.mergedLabelDF[trainLabelColumn].sort_values().unique().shape[0])
        print('newLabels:',self.newLabels)

        #Label replacement
        print("Creating new column for training")
        trainDF[newColName] = trainDF[trainLabelColumn].map(self.__replaceLabel)
        if (validationDF is not None):
            print("Creating new column for validation")
            validationDF[newColName] = validationDF[validationLabelColumn].map(self.__replaceLabel)
        print("===========Transform labels completed")


        if (validationDF is None):
            return trainDF
        else:
            return trainDF, validationDF

    def sanityIsAllProductsExistsInTrainset(self):

        pass