import sys
sys.path.append('../')

import json
import warnings

from utils.read_data import load_data
from utils.smiles2fing import Smiles2Fing
from utils.common import (
    data_split,
    ParameterGrid,
    CV
)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score
)

warnings.filterwarnings('ignore')

x, y = load_data(inhale_type = 'gas')

params_dict = {
    'C': [1, 10, 50],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
params = ParameterGrid(params_dict)


x_train, y_train, x_test, y_test = data_split(x, y, seed = 0)

a = CV(x_train, y_train, LogisticRegression, params[0], seed = 0)
a


result = {}
result['precision'] = []
result['precision'].append({
    params[0]: [a['val_precision'], 
                a['val_recall']]
})

# pd.DataFrame([a])
model = LogisticRegression(random_state = 0, **params[0])
model.fit(x,  y)


