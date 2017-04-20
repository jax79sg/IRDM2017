from HomeDepotCSVReader import HomeDepotReader
from FeatureEngineering import HomeDepotFeature
from DataPreprocessing import DataPreprocessing
import pandas as pd
import numpy as np
# import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import *
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from Evaluation import NDCG_Eval
from HomeDepotCSVWriter import HomeDepotCSVWriter

class XGBoostRanker():
    def __init__(self, feature_train_df):
        self.y_parameter = 'relevance'
        self.x_parameter = [
                'tfidf_product_title',
                'tfidf_product_brand',
                'tfidf_product_description',
                'tfidf_attr_json',
                # 'tfidf_expanded_product_title',
                # 'tfidf_expanded_product_brand',
                # 'tfidf_expanded_product_description',
                # 'tfidf_expanded_attr_json',
                'doc2vec_product_title',
                'doc2vec_product_brand',
                'doc2vec_product_description',
                'doc2vec_attr_json',
                'doc2vec_expanded_product_title',
                'doc2vec_expanded_product_brand',
                'doc2vec_expanded_product_description',
                'doc2vec_expanded_attr_json',
                'bm25',
                'bm25expandedquery',
                'bm25description',
                'bm25title',
                'bm25brand',
                'len_product_title',
                'len_product_description',
                'len_brand',
                'len_search_term',
                'brand_exist',
                'color_exist',
                'wm_product_description',
                'wm_product_title',
                'wm_product_brand',
                'wm_attr_json',
                'sense2vec_all_simscore',
                'sense2vec_keeptag_simscore',
                'sense2vec_uidfact_all_simscore',
                'sense2vec_uidfact_keeptag_simscore',
                'sense2vec_all_attr_simscore',
                'sense2vec_keeptag_attr_simscore',
                'sense2vec_uidfact_all_attr_simscore',
                'sense2vec_uidfact_keeptag_attr_simscore',
                'noun_overlap_counts',
                'noun_uniq_overlap_counts',
                'noun_overlap_ratios',
                'product_uid_threshold',
                'pmi',
                'common_w_title',
                'common_w_description',
                'common_words',
                'search_ratio',
                'title_ratio',
                'desc_ratio',
            ]

        columnName = feature_train_df.columns

        for name in columnName:
            if name.find('color1hot') != -1:
                # print("name: " + name)
                self.x_parameter.append(name)

        # print("x_parameter: ", self.x_parameter)

    def train_classifier(self, trainDF):
        y_train = trainDF[self.y_parameter]
        x_train = trainDF[self.x_parameter]

        RMSE = make_scorer(rmse, greater_is_better=False)

        print ("No of features in input matrix: %d" % len(x_train.columns))
        self._model = XGBClassifier()
        self._model.fit(x_train, y_train)

        y_pred = self._model.predict(x_train)

        print("RMSE: ", rmse(y_train, y_pred))

    def train_Regressor(self, trainDF):
        y_train = trainDF[self.y_parameter]
        x_train = trainDF[self.x_parameter]

        y_train = [y/3 for y in y_train]

        # print(y_train[:10])

        RMSE = make_scorer(rmse, greater_is_better=False)

        print ("No of features in input matrix: %d" % len(x_train.columns))
        self._model = XGBRegressor(learning_rate=0.1, silent=True, objective='binary:logistic', nthread=-1, gamma=0.9,
                                   min_child_weight=1, max_delta_step=0, subsample=0.9, colsample_bytree=0.7,
                                   colsample_bylevel=1, reg_alpha=0.0009, reg_lambda=1, scale_pos_weight=1,
                                   base_score=0.5, seed=0, missing=None, max_depth=7, n_estimators=100)

        self._model.fit(x_train, y_train)

        y_pred = self._model.predict(x_train)
        y_pred = [y*3 for y in y_pred]
        y_train = [y*3 for y in y_train]

        # for i in range(30):
        #     print("Gold: %.2f  Pred: %.2f" %(y_train[i], y_pred[i]))

        print("Train RMSE: ", rmse(y_train, y_pred))

        # plot_importance(self._model, max_num_features=15)
        # plt.show()

    def gridSearch_Regressor(self, trainDF):
        y_train = trainDF[self.y_parameter]
        x_train = trainDF[self.x_parameter]

        y_train = [y / 3 for y in y_train]

        ## Setup Grid Search parameter
        param_grid = {
                      # 'max_depth': [5, 6, 7, 8, 9, 10],
                      # 'min_child_weight': [1, 2, 3, 4],
                      'gamma': [0.6, 0.7, 0.8, 0.9, 1.0],
                      # 'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                      # 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                      # 'reg_alpha': [0.0008, 0.0009],
                      # 'learning_rate': [0.11, 0.1, 0.09],
                      # 'n_estimators': [50, 100, 200],
                      }

        ind_params = {'n_estimators': 100,
                      'seed': 0,
                      'objective': 'binary:logistic',
                      'base_score': 0.5,
                      'colsample_bylevel': 1,
                      'max_delta_step': 0,
                      'missing': None,
                      'reg_lambda': 1,
                      'scale_pos_weight': 1,
                      'silent': True,
                      'learning_rate': 0.1,
                      'max_depth': 9,
                      'min_child_weight': 1,
                      # 'gamma': 0.9,
                      'subsample': 0.9,
                      'colsample_bytree': 0.7,
                      'reg_alpha': 0.0009,
                      }

        RMSE = make_scorer(rmse, greater_is_better=False)

        self._model = GridSearchCV(XGBRegressor(**ind_params),
                                     param_grid=param_grid,
                                     scoring=RMSE,
                                     cv=5,
                                     n_jobs=-1,
                                     error_score='raise')

        self._model.fit(x_train, y_train)

        print("Best parameters found by grid search:")
        print(self._model.best_params_)
        print("Best CV score:")
        print(self._model.best_score_)



    def test_Model(self, test_df, dataname):
        y_test = test_df[self.y_parameter]
        x_test = test_df[self.x_parameter]

        y_test = [y/3 for y in y_test]

        RMSE = make_scorer(rmse, greater_is_better=False)

        y_pred = self._model.predict(x_test)
        y_pred = [y*3 for y in y_pred]
        y_test = [y*3 for y in y_test]

        # for i in range(30):
        #     print("Gold: %.2f  Pred: %.2f" %(y_test[i], y_pred[i]))

        print(dataname + " RMSE: ", rmse(y_test, y_pred))

        # print(test_df.columns)

        result_df = pd.DataFrame()
        result_df['search_term'] = test_df['search_term']
        result_df['product_uid'] = test_df['product_uid']
        result_df['id'] = test_df['id']
        result_df['relevance_int'] = y_pred
        result_df['relevance'] = y_pred

        return result_df



    def gridSearch_classifier(self, trainDF):
        y_train = trainDF[self.y_parameter]
        x_train = trainDF[self.x_parameter]

        ## Setup Grid Search parameter
        param_grid = {
                      # 'max_depth': range(3, 7, 1),
                      # 'min_child_weight': range(1, 4, 1),
                      'gamma': [i / 10.0 for i in range(0, 5)],
                      # 'subsample': [1.0],
                      # 'learning_rate': [0.15, 0.1, 0.05],
                      }

        ind_params = {'n_estimators': 100,
                      'seed': 0,
                      'colsample_bytree': 0.8,
                      'objective': 'binary:logistic',
                      'base_score': 0.5,
                      'colsample_bylevel': 1,
                      # 'gamma': 0,
                      'max_delta_step': 0,
                      'missing': None,
                      'reg_alpha': 0,
                      'reg_lambda': 1,
                      'scale_pos_weight': 1,
                      'silent': True,
                      'learning_rate': 0.05,
                      'max_depth': 4,
                      'min_child_weight': 3,
                      }

        RMSE = make_scorer(rmse, greater_is_better=False)

        model = GridSearchCV(XGBClassifier(**ind_params),
                                     param_grid=param_grid,
                                     # scoring='accuracy',
                                     scoring=RMSE,
                                     cv=5,
                                     n_jobs=-1,
                                     error_score='raise')

        model.fit(x_train, y_train.values)

        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)


def rmse(y_gold, y_pred):
    return mean_squared_error(y_gold, y_pred) ** 0.5


if __name__ == "__main__":
    reader = HomeDepotReader()
    feature_df = reader.getBasicDataFrame("../data/features_final_20170419.csv")

    columnname = feature_df.columns
    feature_train_df = feature_df

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

    print("####  Running: XGBoostRanker.runXGBoostRanker() ####")
    xgb = XGBoostRanker(feature_train_df)
    xgb.train_Regressor(feature_train_df)

    result_public_df = xgb.test_Model(test_public_df, "Public")
    result_private_df = xgb.test_Model(test_private_df, "Private")

    # gold_df = pd.DataFrame()
    # gold_df['search_term'] = test_public_df['search_term']
    # gold_df['product_uid'] = test_public_df['product_uid']
    # gold_df['relevance_int'] = test_public_df['relevance']
    # ndcg = NDCG_Eval()
    # ndcg.computeAvgNDCG(gold_df, result_public_df, "../data/ndcg.csv")
    #
    # gold_df = pd.DataFrame()
    # gold_df['search_term'] = test_private_df['search_term']
    # gold_df['product_uid'] = test_private_df['product_uid']
    # gold_df['relevance_int'] = test_private_df['relevance']
    # ndcg = NDCG_Eval()
    # ndcg.computeAvgNDCG(gold_df, result_private_df, "../data/ndcg.csv")

    result_public_df.pop('product_uid')
    result_public_df.pop('search_term')
    result_public_df.pop('relevance_int')
    HomeDepotCSVWriter().dumpCSV(result_public_df, "../data/xgboost_public.csv")

    result_private_df.pop('product_uid')
    result_private_df.pop('search_term')
    result_private_df.pop('relevance_int')
    HomeDepotCSVWriter().dumpCSV(result_private_df, "../data/xgboost_private.csv")