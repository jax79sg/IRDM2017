from HomeDepotCSVReader import HomeDepotReader
from FeatureEngineering import HomeDepotFeature
import pandas as pd
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
                'doc2vec_product_title', #0.8139
                # 'doc2vec_product_brand',
                # 'doc2vec_product_description',
                # 'doc2vec_attr_json',
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
            ]

    def train(self, trainDF):
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
                                   min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=3,
                                   colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                                   base_score=0.5, seed=0, missing=None)

        self._model.fit(x_train, y_train)

        y_pred = self._model.predict(x_train)

        print("RMSE: ", rmse(y_train, y_pred))


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
    # feature_df = reader.getBasicDataFrame("../data/features.csv")
    # feature_df = reader.getBasicDataFrame("../data/features_Doc2Vec.csv")
    feature_df = reader.getBasicDataFrame("../data/features_Doc2Vec_retrain.csv")

    print("####  Running: XGBoostRanker.runXGBoostRanker() ####")
    xgb = XGBoostRanker()
    xgb.train(feature_df)
    # xgb.train_Regressor(feature_df)
    # xgb.gridSearch(feature_df)

