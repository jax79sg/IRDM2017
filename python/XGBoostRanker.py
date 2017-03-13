from HomeDepotCSVReader import HomeDepotReader
from FeatureEngineering import HomeDepotFeature
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer


class XGBoostRanker():
    def train(self, trainDF):
        y_train = trainDF['relevance']
        x_train = trainDF[[#tfidf_product_title',
                            # 'tfidf_product_description',
                            # 'tfidf_brand',
                            'len_product_title',
                            'len_product_description',
                            # 'len_brand',
                            'len_search_term',
                            'tfidf_product_title',
                            'tfidf_product_description'
                            ]]

        ## Setup Grid Search parameter
        param_grid = {'max_depth': [4, 5],
                      'min_child_weight': [2, 3],
                      'subsample': [0.9, 1.0],
                      'learning_rate': [0.25]
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

        model = GridSearchCV(xgb.XGBClassifier(**ind_params),
                                     param_grid=param_grid,
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





