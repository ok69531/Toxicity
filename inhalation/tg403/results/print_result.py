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
    return sem(values).round(5)


def metric_mean(data, metric: str):
    _ = list(map(lambda x: mean(x[1]), data[metric].items()))
    return _


def metric_se(data, metric: str):
    _ = list(map(lambda x: se(x[1]), data[metric].items()))
    return _


def print_result(inhale_type: str, model: str):
    with open('test_results/' + inhale_type + '_' + model + '.json', 'r') as file:
        d = json.load(file)
    
    result_df = pd.DataFrame({'params': d['model'].values(),
                              'precision_mean': metric_mean(d, 'precision'),
                              'precision_se': metric_se(d, 'precision'),
                              'recall_mean': metric_mean(d, 'recall'),
                              'recall_se': metric_se(d, 'recall'),
                              'accuracy_mean': metric_mean(d, 'accuracy'),
                              'accuracy_se': metric_se(d, 'accuracy'),
                              'f1_mean': metric_mean(d, 'f1'),
                              'f1_se': metric_se(d, 'f1')
                })
    
    return result_df


def print_metrics(df: pd.DataFrame, metric: str):
    idx = df[metric + '_mean'].idxmax()
    return df.iloc[idx]


def model_save(inhale_type: str, model: str, metric: str):
    x, y = load_data(inhale_type)
    result_df = print_result(inhale_type, model)
    best_params = print_metrics(result_df, metric)['params']
    
    if model == 'logistic':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state = 0, **best_params)
        
    elif model == 'dt':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state = 0, **best_params)
    
    elif model == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state = 0, **best_params)
    
    elif model == 'gbt':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(random_state = 0, **best_params)
    
    elif model == 'xgb':
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state = 0, **best_params)
    
    elif model == 'lgb':
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(random_state = 0, **best_params)
    
    elif model == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis(**best_params)
    
    elif model == 'qda':
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        clf = QuadraticDiscriminantAnalysis(**best_params)
    
    elif model == 'plsda':
        from sklearn.cross_decomposition import PLSRegression
        clf = PLSRegression(**best_params)
        
    elif model == 'mlp':
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(random_state = 0, **best_params)
    
    
    if clf == sklearn.cross_decomposition._pls.PLSRegression:
        onehot_y = pd.get_dummies(y)
            
        clf.fit(x, onehot_y)
            
    else:
        clf.fit(x, y)
    
    with open('saved_model/' + inhale_type + '_' + model + '.pkl', 'wb') as file:
        pickle.dump(clf, file)


if __name__ == '__main__':
    inhale = ['vapour', 'aerosol', 'gas']
    metrics = ['precision', 'recall', 'accuracy', 'f1']
    models = ['logistic', 'dt', 'rf', 'gbt', 'xgb', 'lgb', 'lda', 'qda', 'plsda', 'mlp']
    
    comb = list(itertools.product(inhale, models))
    
    for x in tqdm(comb):
        try:
            model_save(x[0], x[1], 'f1')
        except FileNotFoundError:
            print(x[0] + '_' + x[1] + '.json file is empty !')


# if __name__ == '__main__':
#     df_ = print_result('vapour', 'lgb')
#     print(print_metrics(df_, 'f1'))
#     # print(print_metrics(df_, 'f1')[['precision_mean', 'recall_mean', 'accuracy_mean', 'f1_mean']])
