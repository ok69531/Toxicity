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

from sklearn.neural_network import MLPRegressor

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
        'hidden_layer_sizes': [(50), (100, 50, 10), (100, 70, 50, 30, 10)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [50, 100, 200, 300, 500]
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
        
        seed_list = []
        for seed_ in range(10):
            try:
                x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, random_state = seed_)
                
                try:
                    model = MLPRegressor(random_state = seed_, **params[p])
                except: 
                    model = MLPRegressor(**params[p])
                    
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
                
                result['mse']['model'+str(p)].append(mean_squared_error(y_test, pred))
                result['rmse']['model'+str(p)].append(mean_squared_error(y_test, pred)**0.5)
                result['mae']['model'+str(p)].append(mean_absolute_error(y_test, pred))
                
                seed_list.append(seed_)
                if len(seed_list) == 10: break
            
            except:
                pass
            
        
    json.dump(result, open('../results/test_results/mlp.json', 'w'))


if __name__ == '__main__':
    main()

