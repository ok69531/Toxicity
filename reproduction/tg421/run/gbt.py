import sys
sys.path.append('../')

import json
import openpyxl
import warnings
import argparse
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from utils.read_data import load_data
from utils.smiles2fing import Smiles2Fing
from utils.common import ParameterGrid

from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admin-type', type = str, default = 'oral')
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])
        
    x, y = load_data(args.admin_type)

    params_dict = {
        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [5, 10, 50, 100, 130],
        'subsample': [1, 0.8, 0.5],
        'max_depth': [2, 3, 5],
        'min_samples_leaf': [1, 2]
    }
    params = ParameterGrid(params_dict)

    result = {}
    result['model'] = {}
    result['mse'] = {}
    result['rmse'] = {}
    result['mae'] = {}

    for p in tqdm(range(len(params))):
        
        result['model']['model'+str(p)] = params[p]
        result['mse']['model'+str(p)] = []
        result['rmse']['model'+str(p)] = []
        result['mae']['model'+str(p)] = []
        
        for seed_ in range(10):
            x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, random_state = seed_)
            
            try:
                model = GradientBoostingRegressor(random_state = seed_, **params[p])
            except: 
                model = GradientBoostingRegressor(**params[p])
                
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            
            result['mse']['model'+str(p)].append(mean_squared_error(y_test, pred))
            result['rmse']['model'+str(p)].append(mean_squared_error(y_test, pred)**0.5)
            result['mae']['model'+str(p)].append(mean_absolute_error(y_test, pred))
            
        
    json.dump(result, open('../results/test_results/gbt.json', 'w'))


if __name__ == '__main__':
    main()

