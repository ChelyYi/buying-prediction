# -*- coding: UTF-8 -*-

# ©copyright Chely_Yi
# github: https://github.com/ChelyYi

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('./cache/train.txt', sep=' ')
target = 'label'
IDcol = ['u_id','p_id']
predictors = [x for x in train.columns if x not in [target, IDcol]]

def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    print("Fitting... ")
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
    # Predict training set:
    print("Predicting...")
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % roc_auc_score(dtrain[target], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

def parameter():
    # Choose all predictors except target & IDcols
    predictors = [x for x in train.columns if x not in [target, IDcol]]
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=1,
        seed=27)
    modelfit(xgb1, train, predictors)

#parameter()


def para_test1():
    param_test1 = {
        'max_depth': [3, 5, 7,9],
        'min_child_weight': [1, 3, 5]
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                    n_estimators=140,
                                                    max_depth=5,
                                                    min_child_weight=1,
                                                    gamma=0,
                                                    subsample=0.8,
                                                    colsample_bytree=0.8,
                                                    objective='binary:logistic',
                                                    scale_pos_weight=1,
                                                    seed=27
                                                    ),
                            param_grid=param_test1,scoring='roc_auc',n_jobs=4, iid=False,cv=5)
    gsearch1.fit(train[predictors],train[target])
    print(gsearch1.grid_scores_)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)

para_test1()


