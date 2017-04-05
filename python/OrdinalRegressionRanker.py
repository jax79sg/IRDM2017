import mord
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, average_precision_score,recall_score,precision_score,f1_score, accuracy_score
from UserException import ModelNotTrainedException
from DataPreprocessing import DataPreprocessing
from math import sqrt
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import make_scorer


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
            # Best Score: -0.4885966615485714
            # Best Param: {'alpha': 0.02, 'solver': 'sag', 'max_iter': 100000, 'fit_intercept': True, 'copy_X': True, 'tol': 0.01, 'normalize': True}

            # self.model = mord.OrdinalRidge(alpha=1,fit_intercept=True,normalize=False,copy_X=True,max_iter=None,tol=0.001,solver='auto')
            self.model = mord.OrdinalRidge(alpha=0.0001, fit_intercept=True, normalize=False, copy_X=True, max_iter=300000,
                                           tol=0.00001, solver='sag')
            # self.model = mord.OrdinalRidge(alpha=0.00001, fit_intercept=True, normalize=True, copy_X=True, max_iter=1000000,tol=0.0000001, solver='auto')
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
        # print("Generating new labels...")
        # dp=DataPreprocessing()
        # trainDF,validateDF=dp.transformLabels(trainDF=trainDF,validationDF=validateDF, newColName=self.yColDiscrete)
        # print("New labels generated...")

        print("Remove non trainable features...")
        self.xTrain=trainDF
        self.yTrain=trainDF[self.yColDiscrete]
        # self.xValidate=validateDF
        # self.yValidate=validateDF[self.yColDiscrete]

        # self.xTrain.drop('search_term', axis=1, inplace=True)
        # self.xTrain.drop('relevance', axis=1, inplace=True)
        if ('relevance_int' in self.xTrain):
            self.xTrain=self.xTrain.drop('relevance_int', axis=1)
        self.xTrain=self.xTrain.replace('inf',99999)
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
        self.fittedModel=self.model.fit(self.xTrain,self.yTrain)
        self.yPred=self.fittedModel.predict(self.xTrain)
        # print("self.yPred:", self.yPred.shape, self.yPred[1:50, ])

        # print("self.yPred:", list(self.yPred))

        print("Converting to old labels")
        dp=DataPreprocessing()
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
        # self.yPred=self.fittedModel.predict(self.xValidate)
        # # print("self.yPred:", list(self.yPred))
        #
        #
        # print("Converting to old labels")
        # dp=DataPreprocessing()
        # self.yValidate=dp.transformNewLabelToOld(self.yValidate.as_matrix())
        # self.yPred = dp.transformNewLabelToOld(self.yPred)
        # # print("self.yValidate:", self.yValidate.shape,self.yValidate[1:5,])
        # # print("self.yPred:", self.yPred.shape, self.yPred[1:5, ])
        #
        # print("MSE:", mean_squared_error(self.yValidate, self.yPred))
        # print("RMSE:", sqrt(mean_squared_error(self.yValidate, self.yPred)))
        # # print("Accuracy:", accuracy_score(self.yValidate, self.yPred))
        # # print("Precision:", precision_score(self.yValidate, self.yPred, average='micro'))
        # # print("Recall:", recall_score(self.yValidate, self.yPred, average='micro'))
        # # print("F1:", f1_score(self.yValidate, self.yPred, average='micro'))
        # print("+++++++++++++++++++++Validation completed")


    def validate(self,testDF):
        print("+++++++++++++++++++++Validation start")
        print("Remove non trainable features...")
        self.xTest=testDF
        self.yTest=testDF[self.yColDiscrete]
        if ('relevance_int' in self.xTest):
            self.xTest=self.xTest.drop('relevance_int', axis=1)
        self.xTest=self.xTest.replace('inf',99999)
        self.yPred=self.fittedModel.predict(self.xTest)
        print("Converting to old labels")
        dp=DataPreprocessing()
        self.yTest=dp.transformNewLabelToOld(self.yTest)
        self.yPred = dp.transformNewLabelToOld(self.yPred)
        # print("self.yTest:", self.yTest.shape,self.yTest[1:50,])
        # print("self.yPred:", self.yPred.shape, self.yPred[1:50, ])

        print("MSE:", mean_squared_error(self.yTest, self.yPred))
        print("RMSE:", sqrt(mean_squared_error(self.yTest, self.yPred)))
        print("+++++++++++++++++++++Validation end")




    def rank(self,testDf):
        """
        :param testDf:
        :return:
        """
        if(self.fittedModel is None):
            raise ModelNotTrainedException("Model not trained","Please make sure to run train(trainDF,validateDF)")
        else:
            xTest=testDf
            # xTest = testDf[self.xCol]
            self.yPred=self.fittedModel.predict(xTest)



    def gridSearch(self,trainDF,validateDF):
        """
        alpha : {float, array-like}, shape (n_targets)
Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization. Alpha corresponds to C^-1 in other linear models such as LogisticRegression or LinearSVC. If an array is passed, penalties are assumed to be specific to the targets. Hence they must correspond in number.
copy_X : boolean, optional, default True
If True, X will be copied; else, it may be overwritten.
fit_intercept : boolean
Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered).
max_iter : int, optional
Maximum number of iterations for conjugate gradient solver. For ‘sparse_cg’ and ‘lsqr’ solvers, the default value is determined by scipy.sparse.linalg. For ‘sag’ solver, the default value is 1000.
normalize : boolean, optional, default False
If True, the regressors X will be normalized before regression. This parameter is ignored when fit_intercept is set to False. When the regressors are normalized, note that this makes the hyperparameters learnt more robust and almost independent of the number of samples. The same property is not valid for standardized data. However, if you wish to standardize, please use preprocessing.StandardScaler before calling fit on an estimator with normalize=False.
solver : {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’}
Solver to use in the computational routines:
‘auto’ chooses the solver automatically based on the type of data.
‘svd’ uses a Singular Value Decomposition of X to compute the Ridge coefficients. More stable for singular matrices than ‘cholesky’.
‘cholesky’ uses the standard scipy.linalg.solve function to obtain a closed-form solution.
‘sparse_cg’ uses the conjugate gradient solver as found in scipy.sparse.linalg.cg. As an iterative algorithm, this solver is more appropriate than ‘cholesky’ for large-scale data (possibility to set tol and max_iter).
‘lsqr’ uses the dedicated regularized least-squares routine scipy.sparse.linalg.lsqr. It is the fastest but may not be available in old scipy versions. It also uses an iterative procedure.
‘sag’ uses a Stochastic Average Gradient descent. It also uses an iterative procedure, and is often faster than other solvers when both n_samples and n_features are large. Note that ‘sag’ fast convergence is only guaranteed on features with approximately the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.
All last four solvers support both dense and sparse data. However, only ‘sag’ supports sparse input when fit_intercept is True.
New in version 0.17: Stochastic Average Gradient descent solver.
tol : float
Precision of the solution.
        :param trainDF: 
        :param validateDF: 
        :return: 
        """
        param_grid = [{

                            #alpha=1,fit_intercept=True,normalize=False,copy_X=True,max_iter=None,tol=0.001,solver='auto'
                          'alpha': [0.0001,0.0005,0.001,0.0015,0.002,0.005,0.01,0.015,0.02,0.05,0.1,0.5,1],
                          'fit_intercept': [True,False],
                          'normalize': [False,True],
                          'copy_X': [True,False],
                          'max_iter':[100000],
                            'tol':[0.001,0.01],
                        'solver':['sag','lsqr','sparse_cg','cholesky','svd']
        }
                      ]

        def oldLabelScorer(y, y_pred):
            # Custom scoring
            print("Converting to old labels")
            dp = DataPreprocessing()
            y = dp.transformNewLabelToOld(y.as_matrix())
            y_pred = dp.transformNewLabelToOld(y_pred)
            # print("y:", y.shape, y[1:50, ])
            # print("y_pred:", y_pred.shape, y_pred[1:50, ])
            rmse = sqrt(mean_squared_error(y, y_pred))
            # print("RMSE:", rmse)
            return -rmse

        oldLabelScoring = make_scorer(oldLabelScorer, greater_is_better=True)
        optimized_LR = GridSearchCV(mord.OrdinalRidge(),
                                     param_grid=param_grid,
                                     scoring=oldLabelScoring,
                                     cv=5,
                                     # n_jobs=-1,
                                     error_score=0,
                                    verbose=1)
        print("+++++++++++++++++++++GridSearch Training model...")
        print("Remove non trainable features...")
        self.xTrain=trainDF
        self.yTrain=trainDF[self.yColDiscrete]
        if ('relevance_int' in self.xTrain):
            self.xTrain=self.xTrain.drop('relevance_int', axis=1)
        print("+++++++++++++++++++++Training in progress")
        # print("self.xTrain:",list(self.xTrain))
        # print("self.yTrain:", list(self.yTrain))
        self._model = optimized_LR.fit(self.xTrain,self.yTrain)
        print("Training complete")

        print("Best Score: ", optimized_LR.best_score_)
        print("Best Param: ", optimized_LR.best_params_)




if __name__ == "__main__":
    #This part skips the feature training and simply use it.

    print("Reading feature set")
    all_df=pd.read_csv('../data/features_full.csv')
    feature_train_df = all_df[:74067]
    # Must drop these columns for OrdinalRegression
    feature_train_df.drop('id', axis=1, inplace=True)
    feature_train_df.drop('search_term', axis=1, inplace=True)
    feature_train_df.drop('product_uid', axis=1, inplace=True)
    feature_train_df.drop('relevance', axis=1, inplace=True)
    feature_train_df.drop('product_idx', axis=1, inplace=True)
    feature_train_df.drop('Word2VecQueryExpansion', axis=1, inplace=True)
    feature_train_df.drop('doc2vec_search_term_vector', axis=1, inplace=True)
    feature_train_df.drop('doc2vec_product_title_vector', axis=1, inplace=True)
    feature_train_df.drop('doc2vec_product_brand_vector', axis=1, inplace=True)
    feature_train_df.drop('doc2vec_product_description_vector', axis=1, inplace=True)
    feature_train_df.drop('doc2vec_attr_json_vector', axis=1, inplace=True)
    feature_train_df.drop('doc2vec_Word2VecQueryExpansion_vector', axis=1, inplace=True)




    feature_test_df = all_df[74067:]
    feature_test_df.drop('relevance', axis=1, inplace=True)
    #Featuers to play with.
    # all_df.drop('tfidf_product_title', axis=1, inplace=True)
    # all_df.drop('tfidf_product_brand', axis=1, inplace=True)
    # all_df.drop('tfidf_product_description', axis=1, inplace=True)
    # all_df.drop('tfidf_attr_json', axis=1, inplace=True)
    # all_df.drop('tfidf_expanded_product_title', axis=1, inplace=True)
    # all_df.drop('tfidf_expanded_product_brand', axis=1, inplace=True)
    # all_df.drop('tfidf_expanded_product_description', axis=1, inplace=True)
    # all_df.drop('tfidf_expanded_attr_json', axis=1, inplace=True)
    # all_df.drop('doc2vec_product_title', axis=1, inplace=True)
    # all_df.drop('doc2vec_product_brand', axis=1, inplace=True)
    # all_df.drop('doc2vec_product_description', axis=1, inplace=True)
    # all_df.drop('doc2vec_attr_json', axis=1, inplace=True)
    # all_df.drop('doc2vec_expanded_product_title', axis=1, inplace=True)
    # all_df.drop('doc2vec_expanded_product_brand', axis=1, inplace=True)
    # all_df.drop('doc2vec_expanded_product_description', axis=1, inplace=True)
    # all_df.drop('doc2vec_expanded_attr_json', axis=1, inplace=True)
    # all_df.drop('bm25', axis=1, inplace=True)
    # all_df.drop('bm25expandedquery', axis=1, inplace=True)
    # all_df.drop('len_product_title', axis=1, inplace=True)
    # all_df.drop('len_product_description', axis=1, inplace=True)
    # all_df.drop('len_brand', axis=1, inplace=True)
    # all_df.drop('len_search_term', axis=1, inplace=True)
    # all_df.drop('color_exist', axis=1, inplace=True)
    # all_df.drop('brand_exist', axis=1, inplace=True)
    # all_df.drop('color1hot_almond', axis=1, inplace=True)
    # all_df.drop('color1hot_aluminum', axis=1, inplace=True)
    # all_df.drop('color1hot_beige', axis=1, inplace=True)
    # all_df.drop('color1hot_bisque', axis=1, inplace=True)
    # all_df.drop('color1hot_biscuit', axis=1, inplace=True)
    # all_df.drop('color1hot_black', axis=1, inplace=True)
    # all_df.drop('color1hot_blue', axis=1, inplace=True)
    # all_df.drop('color1hot_bone', axis=1, inplace=True)
    # all_df.drop('color1hot_brass', axis=1, inplace=True)
    # all_df.drop('color1hot_bronze', axis=1, inplace=True)
    # all_df.drop('color1hot_brown', axis=1, inplace=True)
    # all_df.drop('color1hot_cedar', axis=1, inplace=True)
    # all_df.drop('color1hot_charcoal', axis=1, inplace=True)
    # all_df.drop('color1hot_cherry', axis=1, inplace=True)
    # all_df.drop('color1hot_chestnut', axis=1, inplace=True)
    # all_df.drop('color1hot_chrome', axis=1, inplace=True)
    # all_df.drop('color1hot_clear', axis=1, inplace=True)
    # all_df.drop('color1hot_color', axis=1, inplace=True)
    # all_df.drop('color1hot_concrete', axis=1, inplace=True)
    # all_df.drop('color1hot_copper', axis=1, inplace=True)
    # all_df.drop('color1hot_cream', axis=1, inplace=True)
    # all_df.drop('color1hot_daylight', axis=1, inplace=True)
    # all_df.drop('color1hot_espresso', axis=1, inplace=True)
    # all_df.drop('color1hot_gold', axis=1, inplace=True)
    # all_df.drop('color1hot_gray', axis=1, inplace=True)
    # all_df.drop('color1hot_green', axis=1, inplace=True)
    # all_df.drop('color1hot_grey', axis=1, inplace=True)
    # all_df.drop('color1hot_ivory', axis=1, inplace=True)
    # all_df.drop('color1hot_java', axis=1, inplace=True)
    # all_df.drop('color1hot_linen', axis=1, inplace=True)
    # all_df.drop('color1hot_mahogany', axis=1, inplace=True)
    # all_df.drop('color1hot_metallic', axis=1, inplace=True)
    # all_df.drop('color1hot_mocha', axis=1, inplace=True)
    # all_df.drop('color1hot_multi', axis=1, inplace=True)
    # all_df.drop('color1hot_natural', axis=1, inplace=True)
    # all_df.drop('color1hot_nickel', axis=1, inplace=True)
    # all_df.drop('color1hot_oak', axis=1, inplace=True)
    # all_df.drop('color1hot_orange', axis=1, inplace=True)
    # all_df.drop('color1hot_pewter', axis=1, inplace=True)
    # all_df.drop('color1hot_pink', axis=1, inplace=True)
    # all_df.drop('color1hot_platinum', axis=1, inplace=True)
    # all_df.drop('color1hot_primed', axis=1, inplace=True)
    # all_df.drop('color1hot_purple', axis=1, inplace=True)
    # all_df.drop('color1hot_red', axis=1, inplace=True)
    # all_df.drop('color1hot_sand', axis=1, inplace=True)
    # all_df.drop('color1hot_silver', axis=1, inplace=True)
    # all_df.drop('color1hot_slate', axis=1, inplace=True)
    # all_df.drop('color1hot_stainless', axis=1, inplace=True)
    # all_df.drop('color1hot_steel', axis=1, inplace=True)
    # all_df.drop('color1hot_tan', axis=1, inplace=True)
    # all_df.drop('color1hot_teal', axis=1, inplace=True)
    # all_df.drop('color1hot_unfinished', axis=1, inplace=True)
    # all_df.drop('color1hot_walnut', axis=1, inplace=True)
    # all_df.drop('color1hot_white', axis=1, inplace=True)
    # all_df.drop('color1hot_wood', axis=1, inplace=True)
    # all_df.drop('color1hot_yellow', axis=1, inplace=True)


    # Run personal models from this point onward
    # runOrdinalRegressionRankerLAD(all_df, None)
    # runOrdinalRegressionRankerOrdRidgeGridSearch(all_df, None)
    # runFacMachineRanker(all_df, None)
    # orModel=runOrdinalRegressionRankerOrdRidge(feature_train_df, None)
    # runLogisticRegressionRanker(all_df, None)
    # runOrdinalRegressionRankerLogit(all_df, None)
    # runOrdinalRegressionRankerLogat(all_df, None)

    print("####  Running: OrdinalRegression ordridge training ####")
    # dp=DataPreprocessing()
    # trainDF,validateDF=dp.generateValidationSet(train_df)
    orModel = OrdinalRegressionRanker('ordridge')
    orModel.train(feature_train_df, None)
    print("####  Completed: OrdinalRegression ordridge training ####")


    #Validation/Test set
    soln_filename = '../data/solution.csv'
    soln_df = pd.read_csv(soln_filename, delimiter=',', low_memory=False, encoding="ISO-8859-1")
    print(soln_df.info())
    dp = DataPreprocessing()
    # df_a.merge(df_b, on='mukey', how='left')
    test_private_df = dp.getGoldTestSet(feature_test_df, soln_df,
                                        testsetoption='Private')  # ,savepath='../data/test_private_gold.csv')
    test_public_df = dp.getGoldTestSet(feature_test_df, soln_df,
                                       testsetoption='Public')  # savepath='../data/test_public_gold.csv')

    test_private_df=dp.transformLabels(test_private_df)
    test_public_df = dp.transformLabels(test_public_df)

    test_private_df.drop('id', axis=1, inplace=True)
    test_private_df.drop('search_term', axis=1, inplace=True)
    test_private_df.drop('product_uid', axis=1, inplace=True)
    test_private_df.drop('relevance', axis=1, inplace=True)
    test_private_df.drop('product_idx', axis=1, inplace=True)
    test_private_df.drop('Word2VecQueryExpansion', axis=1, inplace=True)
    test_private_df.drop('doc2vec_search_term_vector', axis=1, inplace=True)
    test_private_df.drop('doc2vec_product_title_vector', axis=1, inplace=True)
    test_private_df.drop('doc2vec_product_brand_vector', axis=1, inplace=True)
    test_private_df.drop('doc2vec_product_description_vector', axis=1, inplace=True)
    test_private_df.drop('doc2vec_attr_json_vector', axis=1, inplace=True)
    test_private_df.drop('doc2vec_Word2VecQueryExpansion_vector', axis=1, inplace=True)

    test_public_df.drop('id', axis=1, inplace=True)
    test_public_df.drop('search_term', axis=1, inplace=True)
    test_public_df.drop('product_uid', axis=1, inplace=True)
    test_public_df.drop('relevance', axis=1, inplace=True)
    test_public_df.drop('product_idx', axis=1, inplace=True)
    test_public_df.drop('Word2VecQueryExpansion', axis=1, inplace=True)
    test_public_df.drop('doc2vec_search_term_vector', axis=1, inplace=True)
    test_public_df.drop('doc2vec_product_title_vector', axis=1, inplace=True)
    test_public_df.drop('doc2vec_product_brand_vector', axis=1, inplace=True)
    test_public_df.drop('doc2vec_product_description_vector', axis=1, inplace=True)
    test_public_df.drop('doc2vec_attr_json_vector', axis=1, inplace=True)
    test_public_df.drop('doc2vec_Word2VecQueryExpansion_vector', axis=1, inplace=True)

    print(test_public_df)
    print("Validating public testset")
    orModel.validate(test_public_df)

    print("Validating private testset")
    orModel.validate(test_private_df)
