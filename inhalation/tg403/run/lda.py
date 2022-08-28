import sys
sys.path.append('../')

import json
import warnings
import argparse
import numpy as np

from tqdm import tqdm

from utils.read_data import load_data
from utils.smiles2fing import Smiles2Fing
from utils.common import (
    data_split,
    ParameterGrid,
    CV
)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inhale-type", type=str, default="vapour")
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    x, y = load_data(inhale_type = args.inhale_type)

    params_dict1 = {
        'solver': ['lsqr', 'eigen'],
        'shrinkage': np.logspace(-3, 0, 30)
    }
    params_dict2 = {
        'solver': ['svd'],
        'tol': np.logspace(-5, -3, 20)
    }
    params = ParameterGrid(params_dict1)
    params.extend(ParameterGrid(params_dict2))

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
        
        for seed_ in range(10):
            # x_train, y_train, x_test, y_test = data_split(x, y, seed = seed_)
            
            r_ = CV(x, 
                    y, 
                    LinearDiscriminantAnalysis, 
                    params[p], 
                    seed = seed_)
            
            result['precision']['model'+str(p)].append(r_['val_precision'])
            result['recall']['model'+str(p)].append(r_['val_recall'])
            result['f1']['model'+str(p)].append(r_['val_f1'])
            result['accuracy']['model'+str(p)].append(r_['val_accuracy'])
        
    json.dump(result, open('../results/' + args.inhale_type + '_lda.json', 'w'))


if __name__ == '__main__':
    main()
