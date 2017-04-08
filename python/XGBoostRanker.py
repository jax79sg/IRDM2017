from HomeDepotCSVReader import HomeDepotReader
from FeatureEngineering import HomeDepotFeature
from DataPreprocessing import DataPreprocessing
import pandas as pd
import numpy as np
# import xgboost as xgb
from xgboost import *
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer


class XGBoostRanker():
    def __init__(self):
        self.y_parameter = 'relevance'
        self.x_parameter = [
                # 'tfidf_product_title', # 0.6383
                # 'tfidf_product_brand',
                # 'tfidf_product_description',
                # 'tfidf_attr_json',
                # 'tfidf_expanded_product_title',
                # 'tfidf_expanded_product_brand',
                # 'tfidf_expanded_product_description',
                # 'tfidf_expanded_attr_json',
                'doc2vec_product_title', #0.62808
                'doc2vec_product_brand',
                'doc2vec_product_description',
                'doc2vec_attr_json',
                # 'doc2vec_expanded_product_title',
                # 'doc2vec_expanded_product_brand',
                # 'doc2vec_expanded_product_description',
                # 'doc2vec_expanded_attr_json',
                # 'bm25',
                # 'bm25expandedquery',
                # 'len_product_title',
                # 'len_product_description',
                # 'len_brand',
                # 'len_search_term',
                # 'brand_exist',
                # 'color_exist',
                # 'color1hot_almond',
                # 'color1hot_aluminum',
                # 'color1hot_beige',
                # 'color1hot_biscuit',
                # 'color1hot_bisque',
                # 'color1hot_black',
                # 'color1hot_blue',
                # 'color1hot_bone',
                # 'color1hot_brass',
                # 'color1hot_bronze',
                # 'color1hot_brown',
                # 'color1hot_cedar',
                # 'color1hot_charcoal',
                # 'color1hot_cherry',
                # 'color1hot_chestnut',
                # 'color1hot_chrome',
                # 'color1hot_clear',
                # 'color1hot_color',
                # 'color1hot_concrete',
                # 'color1hot_copper',
                # 'color1hot_cream',
                # 'color1hot_daylight',
                # 'color1hot_espresso',
                # 'color1hot_gold',
                # 'color1hot_gray',
                # 'color1hot_green',
                # 'color1hot_grey',
                # 'color1hot_ivory',
                # 'color1hot_java',
                # 'color1hot_linen',
                # 'color1hot_mahogany',
                # 'color1hot_metallic',
                # 'color1hot_mocha',
                # 'color1hot_multi',
                # 'color1hot_natural',
                # 'color1hot_nickel',
                # 'color1hot_oak',
                # 'color1hot_orange',
                # 'color1hot_pewter',
                # 'color1hot_pink',
                # 'color1hot_platinum',
                # 'color1hot_primed',
                # 'color1hot_purple',
                # 'color1hot_red',
                # 'color1hot_sand',
                # 'color1hot_silver',
                # 'color1hot_slate',
                # 'color1hot_stainless',
                # 'color1hot_steel',
                # 'color1hot_tan',
                # 'color1hot_teal',
                # 'color1hot_unfinished',
                # 'color1hot_walnut',
                # 'color1hot_white',
                # 'color1hot_wood',
                # 'color1hot_yellow',
                # 'wm_product_description',
                # 'wm_product_title',
                # 'wm_product_brand',
                # 'wm_attr_json',
            ]

    def train_classifier(self, trainDF):
        y_train = trainDF[self.y_parameter]
        x_train = trainDF[self.x_parameter]

        RMSE = make_scorer(rmse, greater_is_better=False)

        print ("No of features in input matrix: %d" % len(x_train.columns))
        optimised_params = {'eta': 0.1, 'seed':0, 'subsample': 0.55, 'colsample_bytree': 0.8,
                            'objective': 'binary:logistic', 'max_depth':4, 'min_child_weight':3,
                            'learning_rate': 0.05, 'reg_alpha': 0.05, 'scoring':RMSE, 'n_estimators': 1000,
                            'base_score': 0.5}
        # Create DMatrix to make XGBoost more efficient
        # xgdmat = DMatrix(x_train, y_train)
        # self._model = train(optimised_params, xgdmat, num_boost_round=432,  verbose_eval=False)
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
        # optimised_params = {'eta': 0.1, 'seed':0, 'subsample': 0.55, 'colsample_bytree': 0.8,
        #                     'objective': 'binary:logistic', 'max_depth':4, 'min_child_weight':3,
        #                     'learning_rate': 0.05, 'reg_alpha': 0.05, 'scoring':RMSE, 'n_estimators': 1000,
        #                     'base_score': 0.5}
        # # Create DMatrix to make XGBoost more efficient
        # # xgdmat = DMatrix(x_train, y_train)
        # # self._model = train(optimised_params, xgdmat, num_boost_round=432,  verbose_eval=False)
        self._model = XGBRegressor(learning_rate=0.25, silent=False, objective="reg:linear", nthread=-1, gamma=0,
                                   min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                                   colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                                   base_score=0.5, seed=0, missing=None, max_depth=5)

        self._model.fit(x_train, y_train)

        y_pred = self._model.predict(x_train)
        y_pred = [y*3 for y in y_pred]
        y_train = [y*3 for y in y_train]

        for i in range(30):
            print("Gold: %.2f  Pred: %.2f" %(y_train[i], y_pred[i]))

        print("Train RMSE: ", rmse(y_train, y_pred))

    def test_Model(self, test_df):
        y_test = test_df[self.y_parameter]
        x_test = test_df[self.x_parameter]

        y_test = [y/3 for y in y_test]

        RMSE = make_scorer(rmse, greater_is_better=False)

        y_pred = self._model.predict(x_test)
        y_pred = [y*3 for y in y_pred]
        y_test = [y*3 for y in y_test]

        # for i in range(30):
        #     print("Gold: %.2f  Pred: %.2f" %(y_test[i], y_pred[i]))

        print("Test RMSE: ", rmse(y_test, y_pred))


    def gridSearch(self, trainDF):
        y_train = trainDF[self.y_parameter]
        x_train = trainDF[self.x_parameter]

        ## Setup Grid Search parameter
        param_grid = {'max_depth': [4, 5],
                      'min_child_weight': [3],
                      'subsample': [1.0],
                      'learning_rate': [0.25, 0.1]
                      }

        ind_params = {'n_estimators': 100,
                      'seed': 0,
                      'colsample_bytree': 0.8,
                      'objective': 'binary:logistic',
                      'base_score': 0.5,
                      'colsample_bylevel': 1,
                      'gamma': 0,
                      'max_delta_step': 0,
                      'missing': None,
                      'reg_alpha': 0,
                      'reg_lambda': 1,
                      'scale_pos_weight': 1,
                      'silent': True,
                      }

        RMSE = make_scorer(rmse, greater_is_better=False)

        model = GridSearchCV(XGBClassifier(**ind_params),
                                     param_grid=param_grid,
                                     # scoring='accuracy',
                                     scoring=RMSE,
                                     cv=2,
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
    # feature_df = reader.getBasicDataFrame("../data/features_full2.csv")
    feature_df = reader.getBasicDataFrame("../data/features_doc2vec_dm0_sam6.csv")
    # feature_df = reader.getBasicDataFrame("../data/features_Doc2Vec_retrain.csv")

    print(feature_df.info())
    columnname = feature_df.columns
    feature_train_df = feature_df

    # print(columnname)

    # feature_train_df = feature_df[:74067]
    # feature_test_df = feature_df[74067:]
    #
    # # feature_test_df['relevance'] = np.zeros(166693)
    # # feature_test_df['relevance_int'] = np.zeros(166693)
    # feature_test_df.pop('relevance')
    # print(feature_test_df.info())
    #
    # soln_filename = '../data/solution.csv'
    # soln_df = pd.read_csv(soln_filename, delimiter=',', low_memory=False, encoding="ISO-8859-1")
    # print(soln_df.info())
    # dp = DataPreprocessing()
    # # df_a.merge(df_b, on='mukey', how='left')
    # test_private_df = dp.getGoldTestSet(feature_test_df, soln_df,
    #                                     testsetoption='Private')  # ,savepath='../data/test_private_gold.csv')
    # test_public_df = dp.getGoldTestSet(feature_test_df, soln_df,
    #                                    testsetoption='Public')  # savepath='../data/test_public_gold.csv')

    # print(feature_train_df)



    print("####  Running: XGBoostRanker.runXGBoostRanker() ####")
    xgb = XGBoostRanker()
    # xgb.train(feature_df)
    xgb.train_Regressor(feature_train_df)
    # xgb.gridSearch(feature_df)

    # xgb.test_Model(test_public_df)
    # xgb.test_Model(test_private_df)

