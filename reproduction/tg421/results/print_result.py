import sys
sys.path.append('../')

import json
import pickle
import itertools

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.stats import sem
from utils.read_data import load_data

import sklearn
from sklearn.cross_decomposition import PLSRegression

from rdkit import RDLogger

pd.set_option('mode.chained_assignment', None)
RDLogger.DisableLog('rdApp.*')


def mean(values):
    return np.mean(values).round(3)


def se(values):
    return sem(values).round(3)


def metric_mean(data, metric: str):
    _ = list(map(lambda x: mean(x[1]), data[metric].items()))
    return _


def metric_se(data, metric: str):
    _ = list(map(lambda x: se(x[1]), data[metric].items()))
    return _


def print_result(model: str):
    with open('test_results/' + model + '.json', 'r') as file:
        d = json.load(file)
    
    result_df = pd.DataFrame({'params': d['model'].values(),
                              'mse_mean': metric_mean(d, 'mse'),
                              'mse_se': metric_se(d, 'mse'),
                              'rmse_mean': metric_mean(d, 'rmse'),
                              'rmse_se': metric_se(d, 'rmse'),
                              'mae_mean': metric_mean(d, 'mae'),
                              'mae_se': metric_se(d, 'mae'),
                })
    
    return result_df


def print_metrics(df: pd.DataFrame, metric: str):
    idx = df[metric + '_mean'].idxmax()
    return df.iloc[idx]


# def model_save(model: str, metric: str):
#     x, y = load_data('oral')
#     result_df = print_result(model)
#     best_params = print_metrics(result_df, metric)['params']
    
#     if model == 'logistic':
#         from sklearn.linear_model import LogisticRegression
#         clf = LogisticRegression(random_state = 0, **best_params)
        
#     elif model == 'dt':
#         from sklearn.tree import DecisionTreeClassifier
#         clf = DecisionTreeClassifier(random_state = 0, **best_params)
    
#     elif model == 'rf':
#         from sklearn.ensemble import RandomForestClassifier
#         clf = RandomForestClassifier(random_state = 0, **best_params)
    
#     elif model == 'gbt':
#         from sklearn.ensemble import GradientBoostingClassifier
#         clf = GradientBoostingClassifier(random_state = 0, **best_params)
    
#     elif model == 'xgb':
#         from xgboost import XGBClassifier
#         clf = XGBClassifier(random_state = 0, **best_params)
    
#     elif model == 'lgb':
#         from lightgbm import LGBMClassifier
#         clf = LGBMClassifier(random_state = 0, **best_params)
    
#     elif model == 'lda':
#         from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#         clf = LinearDiscriminantAnalysis(**best_params)
    
#     elif model == 'qda':
#         from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#         clf = QuadraticDiscriminantAnalysis(**best_params)
    
#     elif model == 'plsda':
#         from sklearn.cross_decomposition import PLSRegression
#         clf = PLSRegression(**best_params)
        
#     elif model == 'mlp':
#         from sklearn.neural_network import MLPClassifier
#         clf = MLPClassifier(random_state = 0, **best_params)
    
#     if clf == sklearn.cross_decomposition._pls.PLSRegression:
#         onehot_y = pd.get_dummies(y)
            
#         clf.fit(x, onehot_y)
            
#     else:
#         clf.fit(x, y)
    
#     with open('saved_model/' + model + '.pkl', 'wb') as file:
#         pickle.dump(clf, file)


# if __name__ == '__main__':
#     metrics = ['precision', 'recall', 'accuracy', 'f1']
#     models = ['logistic', 'dt', 'rf', 'gbt', 'xgb', 'lgb', 'lda', 'qda', 'plsda', 'mlp']
    
#     for m in tqdm(models):
#         try:
#             model_save(m, 'f1')
#         except FileNotFoundError:
#             print(m + '.json file is empty !')


# if __name__ == '__main__':
#     models = ['logistic', 'dt', 'rf', 'gbt', 'xgb', 'lgb', 'lda', 'qda', 'plsda', 'mlp']
    
#     for m in tqdm(models):
#         try:
#            df_ = print_result(m)
#            print('\n', m, 
#                  '\n', print_metrics(df_, 'mae'))
#                 #   '\n', print_metrics(df_, 'f1')[['precision_mean', 'recall_mean', 'accuracy_mean', 'f1_mean']])
#         except:
#             print('\n', m, 'result is empty')


# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt
# clf = GradientBoostingRegressor(random_state = 0, **print_metrics(df_, 'mae').params)
# x, y = load_data('oral')
# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, random_state = 0)
# clf.fit(x_train, y_train)

# pred = clf.predict(x_test)

# l = np.linspace(min(y_test), max(y_test))
# plt.figure(figsize = (8, 8))
# plt.plot(l, l, color = 'orange')
# plt.scatter(y_test, pred)
# plt.xlabel('true')
# plt.ylabel('pred')
# plt.show()
# plt.close()
