
from sklearn.grid_search import GridSearchCV

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, average_precision_score,recall_score,precision_score,f1_score, accuracy_score
from math import sqrt
import DataPreprocessing

class LogisticRegressionRanker():
    _model=None
    yColDiscrete = 'relevance_int'

    def train(self, trainDF, svalidateDF):
        self._model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='sag', max_iter=10000, multi_class='multinomial', verbose=1, warm_start=False, n_jobs=-1)
        print("+++++++++++++++++++++Training model...")

        print("Remove non trainable features...")
        self.xTrain=trainDF
        self.yTrain=trainDF[self.yColDiscrete]
        # self.xValidate=validateDF
        # self.yValidate=validateDF[self.yColDiscrete]

        # self.xTrain.drop('search_term', axis=1, inplace=True)
        # self.xTrain.drop('relevance', axis=1, inplace=True)
        if ('relevance_int' in self.xTrain):
            self.xTrain = self.xTrain.drop('relevance_int', axis=1)
        # self.xTrain.drop('product_idx', axis=1, inplace=True)
        # self.xTrain.drop('Word2VecQueryExpansion', axis=1, inplace=True)


        # self.xValidate.drop('search_term', axis=1, inplace=True)
        # self.xValidate.drop('relevance', axis=1, inplace=True)
        # self.xValidate.drop('relevance_int', axis=1, inplace=True)
        # self.xValidate.drop('product_idx', axis=1, inplace=True)
        # self.xValidate.drop('Word2VecQueryExpansion', axis=1, inplace=True)


        print("+++++++++++++++++++++Training in progress")
        # print("self.xTrain:",list(self.xTrain))
        # print("self.yTrain:", list(self.yTrain))
        fittedModel=self._model.fit(self.xTrain,self.yTrain)
        self.yPred=fittedModel.predict(self.xTrain)
        # print("self.yPred:", list(self.yPred))


        print("Converting to old labels")
        dp=DataPreprocessing.DataPreprocessing()
        self.yTrain=dp.transformNewLabelToOld(self.yTrain.as_matrix())
        self.yPred = dp.transformNewLabelToOld(self.yPred)
        print("self.yTrain:", self.yTrain.shape,self.yTrain[1:50,])
        print("self.yPred:", self.yPred.shape, self.yPred[1:50, ])


        print("MSE:", mean_squared_error(self.yTrain, self.yPred))
        print("RMSE:", sqrt(mean_squared_error(self.yTrain, self.yPred)))
        # print("Accuracy:", accuracy_score(self.yTrain, self.yPred))
        # print("Precision:", precision_score(self.yTrain, self.yPred, average='micro'))
        # print("Recall:", recall_score(self.yTrain, self.yPred, average='micro'))
        # print("F1:", f1_score(self.yTrain, self.yPred, average='micro'))
        print("+++++++++++++++++++++Training completed")

        # ##NOTE: This evaluation is not suitable for ranking.. Need to look at DCG, NDCG, MAP instead.
        # print("+++++++++++++++++++++Validation in progress")
        # # print("self.xValidate:", list(self.xValidate))
        # # print("self.yValidate:", list(self.yValidate))
        # self.yPred=fittedModel.predict(self.xValidate)
        # # print("self.yPred:", list(self.yPred))
        #
        # print("Converting to old labels")
        # dp=DataPreprocessing.DataPreprocessing()
        # self.yValidate=dp.transformNewLabelToOld(self.yValidate.as_matrix())
        # self.yPred = dp.transformNewLabelToOld(self.yPred)
        # # print("self.yValidate:", self.yValidate.shape,self.yValidate[1:5,])
        # # print("self.yPred:", self.yPred.shape, self.yPred[1:5, ])
        #
        #
        #
        # print("MSE:", mean_squared_error(self.yValidate, self.yPred))
        # print("RMSE:", sqrt(mean_squared_error(self.yValidate, self.yPred)))
        # # print("Accuracy:", accuracy_score(self.yValidate, self.yPred))
        # # print("Precision:", precision_score(self.yValidate, self.yPred, average='micro'))
        # # print("Recall:", recall_score(self.yValidate, self.yPred, average='micro'))
        # # print("F1:", f1_score(self.yValidate, self.yPred, average='micro'))
        # print("+++++++++++++++++++++Validation completed")

    def validateModel(self, xValidate, yValidate):

        pass

    def gridSearchandCrossValidate(self, xTrain, yTrain):
        ## Setup Grid Search parameter
        # param_grid = [{
        #                   'solver': ['liblinear'],
        #                   'C': [0.15, 0.19, 0.2, 0.21, 0.25],
        #                   'class_weight':[None],  # None is better
        #                   'penalty': ['l2', 'l1'],
        #               }

        param_grid = [{
                            'alpha':[0.0050,0.0015,0.0025,0.004]
                          # 'penalty': ['l1']
                      }

                      #   ,
                      # {
                      #     'solver': ['newton-cg', 'lbfgs', 'sag'],
                      #     'C': [0.1, 0.5, 1.0, 2.0],
                      #     'max_iter':[50000],
                      #     'class_weight': [None],  # None is better
                      #     'penalty': ['l2'],
                      # }
                      ]

        optimized_LR = GridSearchCV(LogisticRegression(),
                                     param_grid=param_grid,
                                     scoring='roc_auc',
                                     cv=3,
                                     n_jobs=-1,
                                     error_score='raise',
                                     verbose=0
                                     )
        print("Grid Searching...")
        self._model = optimized_LR.fit(xTrain, yTrain)
        print("Best Score: ", optimized_LR.best_score_)
        print("Best Param: ", optimized_LR.best_params_)

        scores = optimized_LR.grid_scores_
        for i in range(len(scores)):
            print(optimized_LR.grid_scores_[i])




