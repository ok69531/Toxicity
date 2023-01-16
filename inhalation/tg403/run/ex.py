import sys
sys.path.append('../')

import json
import warnings
import argparse

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score
)

from utils.read_data import load_data
from utils.smiles2fing import Smiles2Fing
from utils.common import ParameterGrid

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inhale-type", type=str, default="vapour")
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    x, y = load_data(inhale_type = args.inhale_type)
    
    test_num = round(len(x) * 0.1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_num, shuffle = True, random_state = 42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = test_num, shuffle = True, random_state = 42)
    
    params_dict = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 1, 2, 3, 4, 5],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3]
    }
    params = ParameterGrid(params_dict)

    
    result = {}
    result['model'] = {}
    result['precision'] = {}
    result['recall'] = {}
    result['f1'] = {}
    result['accuracy'] = {}

    for p in tqdm(range(len(params))):
        
        result['model']['model'+str(p)] = params[p]
        result['precision']['model'+str(p)] = []
        result['recall']['model'+str(p)] = []
        result['f1']['model'+str(p)] = []
        result['accuracy']['model'+str(p)] = []
        
        for seed in range(10):
            
            try:
                model = DecisionTreeClassifier(random_state = seed, **params[p])
            except: 
                model = DecisionTreeClassifier(**params[p])
                
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            
            result['precision']['model'+str(p)].append(precision_score(y_val, pred, average = 'macro'))
            result['recall']['model'+str(p)].append(recall_score(y_val, pred, average = 'macro'))
            result['f1']['model'+str(p)].append(f1_score(y_val, pred, average = 'macro'))
            result['accuracy']['model'+str(p)].append(accuracy_score(y_val, pred))
    
    json.dump(result, open('../results/val_results/' + args.inhale_type + '_dt.json', 'w'))


#%%
# print_metric
# model save

#%%
# predict test 
