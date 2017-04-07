

import numpy as np
from sklearn.metrics import mean_squared_error, average_precision_score,recall_score,precision_score,f1_score, accuracy_score
from sklearn.grid_search import GridSearchCV
from UserException import ModelNotTrainedException
import datetime
import pandas as pd
from fastFM import als
import scipy as scipy

from Utilities import Utility
from  sklearn.metrics import roc_auc_score
from sgdFMClassification import SGDFMClassification
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
import DataPreprocessing
from math import sqrt

class FacMachineRanker():
    xTrain=None
    yTrain=None

    _model=None
    yColDiscrete = 'relevance_int'

    def __predictClickOneProb(self,testDF):
        """
        Perform prediction for click label.
        Take the output of click=1 probability as the CTR.
        :param oneBidRequest:
        :return:
        """

        print("Setting up X test for prediction")
        xTest=testDF

        print("Converting to sparse matrix")
        xTest = scipy.sparse.csc_matrix(xTest.as_matrix())

        # predict click labels for the test set
        print("Predicting test set...")

        # FastFM only give a probabilty of a click=1
        predictedClickOneProb = self._model.predict_proba(xTest)

        return predictedClickOneProb

    def __predictClickOne(self,testDF):
        """
        Perform prediction for click label.
        Take the output of click=0 or 1 as the CTR.
        :param oneBidRequest:
        :return:
        """

        print("Setting up X test for prediction")
        xTest=testDF

        print("Converting to sparse matrix")
        xTest = scipy.sparse.csc_matrix(xTest.as_matrix())

        # predict click labels for the test set
        print("Predicting test set...")

        # FastFM only give a probabilty of a click=1
        predictedClick = self._model.predict(xTest, self.getThreshold())

        return predictedClick

    def train(self,trainDF,validateDF):
        print("+++++++++++++++++++++Training model...")
        print("Remove non trainable features...")
        self.xTrain=trainDF
        self.yTrain=trainDF[self.yColDiscrete]
        if ('relevance_int' in self.xTrain):
            self.xTrain=self.xTrain.drop('relevance_int', axis=1)

        print("OneHot encoding")
        self.xTrain=pd.get_dummies(self.xTrain,sparse=True)
        self.xTrain= scipy.sparse.csc_matrix(self.xTrain)

        fm = SGDFMClassification(n_iter=1000, rank=16, l2_reg_w=0.0005, l2_reg_V=0.0005, l2_reg=0.0005,step_size=0.01)
        self._model = OneVsRestClassifier(fm)

        self.fittedModel=self._model.fit(self.xTrain,self.yTrain)
        self.yPred=self.fittedModel.predict(self.xTrain)

        print("Converting to old labels")
        dp=DataPreprocessing.DataPreprocessing()
        self.yTrain=dp.transformNewLabelToOld(self.yTrain.as_matrix())
        self.yPred = dp.transformNewLabelToOld(self.yPred)
        print("self.yTrain:", self.yTrain.shape,self.yTrain[1:50,])
        print("self.yPred:", self.yPred.shape, self.yPred[1:100, ])

        print("MSE:", mean_squared_error(self.yTrain, self.yPred))
        print("RMSE:", sqrt(mean_squared_error(self.yTrain, self.yPred)))
        print("+++++++++++++++++++++Training completed")



    def gridSearchandCrossValidateFastSGD(self, X,y, retrain=True):
        """
        Perform gridsearch on FM model
        :param X:
        :param y:
        :param retrain:
        :return:
        """
        # n_iter=100000, rank=2, l2_reg_w=0.01, l2_reg_V=0.01, l2_reg=0.01, step_size=0.004
        print("Getting xTrain")
        xTrain = X
        yTrain = y
        print("xTrain:", xTrain.shape,list(xTrain))
        print("yTrain:", yTrain.shape,set(yTrain['click']),"ListL",list(yTrain))
        yTrain['click'] = yTrain['click'].map({0: -1, 1: 1})


        # xTrain.to_csv("data.pruned/xTrain.csv")
        # yTrain.to_csv("data.pruned/yTrain.csv")

        print("xTrain:",list(xTrain))
        xTrain=xTrain.as_matrix()
        yTrain = yTrain['click'].as_matrix()
        print("Performing oversampling to even out")
        xTrain,yTrain=ImbalanceSampling().oversampling_SMOTE(X=xTrain,y=yTrain)

        print("Factorisation Machine with SGD solver will be used for training")
        print("Converting X to sparse matrix, required by FastFM")
        xTrain = scipy.sparse.csc_matrix(xTrain)

        param_grid = [{
                          'n_iter': [150000,200000,250000],
                          'l2_reg_w': [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
                          'l2_reg_V': [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
                          # 'l2_reg': [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
                          'step_size': [0.0005,0.004,0.007],
                          'rank':[32,36,42,46,52,56,64]
                        # 'n_iter': [5000],
                        # 'l2_reg_w': [0.0005, 0.001],
                        # 'l2_reg_V': [0.0005, 0.001],
                        # 'l2_reg': [0.0005],
                        # 'step_size': [ 0.004]

        }
                      ]

        optimized_LR = GridSearchCV(SGDFMClassification(),
                                     param_grid=param_grid,
                                     scoring='roc_auc',
                                     cv=5,
                                     # n_jobs=-1,
                                     error_score='raise',
                                    verbose=1)
        print("Training model..")
        print(datetime.datetime.now())
        if(retrain):
            self._model = optimized_LR.fit(xTrain, yTrain)
        print("Training complete")
        print(datetime.datetime.now())

        print("Best Score: ", optimized_LR.best_score_)
        print("Best Param: ", optimized_LR.best_params_)







