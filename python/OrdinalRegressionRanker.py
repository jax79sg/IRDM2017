import mord
import numpy as np
from sklearn.metrics import mean_squared_error, average_precision_score,recall_score,precision_score,f1_score
from UserException import ModelNotTrainedException
from DataPreprocessing import DataPreprocessing


"""
Changelog:
- 14/3 KS First commit
"""

class OrdinalRegressionRanker(object):

    xTrain=None
    yTrain=None
    xValidate=None
    yValidate=None
    yPred=None
    oldLabels=None
    newLabels=None
    fittedModel=None
    model=None

    yCol='relevance'
    yColDiscrete = 'relevance_int'
    xCol=[#tfidf_product_title',
                            # 'tfidf_product_description',
                            # 'tfidf_brand',
                            'len_product_title',
                            'len_product_description',
                            # 'len_brand',
                            'len_search_term',
                            # 'tfidf_product_title',
                            # 'tfidf_product_description'
                            ]

    def __init__(self,modelType):
        """
        :param modelType: 'logIT','logAT','ordRidge','lad','multiclasslogistic'
        """


        if(modelType.lower()=='logit'):
            print("Using Logistic Immediate-Threshold variant")
            self.model = mord.LogisticIT(alpha=1.0, verbose=0, max_iter=1000)
        elif (modelType.lower()=='logat'):
            print("Using Logistic All-Threshold variant")
            self.model = mord.LogisticAT(alpha=1.0, verbose=0, max_iter=1000)
        elif (modelType.lower()=='ordridge'):
            print("Using Ordinal Ridge variant")
            self.model = mord.OrdinalRidge(alpha=1,fit_intercept=True,normalize=False,copy_X=True,max_iter=None,tol=0.001,solver='auto')
        elif (modelType.lower()=='lad'):
            print("Using Least Absolute Deviation")
            self.model = mord.LAD(epsilon=0.0, tol=0.0001, C=1.0, loss='l1', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000)
        elif (modelType.lower()=='multiclasslogistic'):
            print("Using Multiclass Logistic")
            self.model = mord.MulticlassLogistic(alpha=1.0, verbose=0, maxiter=1000)
        else:
            print("Model selection not recognised.\nDefaulted to Logistic All-Threshold variant")
            self.model = mord.LogisticIT(alpha=1.0, verbose=1, max_iter=1000)


    def rmse(y_gold, y_pred):
        return mean_squared_error(y_gold, y_pred) ** 0.5



    def train(self,trainDF,validateDF):
        print("+++++++++++++++++++++Training model...")
        print("Generating new labels...")
        dp=DataPreprocessing()
        trainDF,validateDF=dp.transformLabels(trainDF=trainDF,validationDF=validateDF, newColName=self.yColDiscrete)
        print("New labels generated...")

        self.xTrain=trainDF[self.xCol]
        self.yTrain=trainDF[self.yColDiscrete]
        self.xValidate=validateDF[self.xCol]
        self.yValidate=validateDF[self.yColDiscrete]

        print("self.xTrain:",self.xTrain.head(1))
        print("self.yTrain:", self.yTrain.head(1))
        print("self.xValidate:", self.xValidate.head(1))
        print("self.yValidate:", self.yValidate.head(1))


        self.fittedModel=self.model.fit(self.xTrain,self.yTrain)
        print("+++++++++++++++++++++Training completed")

        print("+++++++++++++++++++++Validation in progress")
        self.yPred=self.fittedModel.predict(self.xValidate)
        print("Precision:", precision_score(self.yValidate, self.yPred, average='micro'))
        print("Recall:", recall_score(self.yValidate, self.yPred, average='micro'))
        print("F1:", f1_score(self.yValidate, self.yPred, average='micro'))
        print("+++++++++++++++++++++Validation completed")

    def rank(self,testDf):
        if(self.fittedModel is None):
            raise ModelNotTrainedException("Model not trained","Please make sure to run train(trainDF,validateDF)")
        else:
            self.xValidate=testDf[self.xCol]
            self.yValidate = testDf[self.yCol]
            self.yPred=self.fittedModel.predict(testDf[self.xCol])





