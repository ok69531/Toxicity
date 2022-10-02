#%%
import cirpy
import openpyxl
import warnings

import pandas as pd
import numpy as np

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings("ignore")


#%%
'''
    data split
'''

data = pd.read_excel('tg420_ld50.xlsx')


len(data['CasRN'].unique())

data['unit'].unique()
data['unit'].isna().sum()
# data = data[data['unit'].notna()]
data = data[data['lower_value'].notna()]

casrn_na_idx = data[data['CasRN'] == '-'].index

data = data.drop(casrn_na_idx).reset_index(drop = True)


#%%
def unify(unit, value):
    if unit == 'mg/kg':
        v_ = value
    
    elif unit == 'g/kg':
        v_ = value * 1000
    
    return v_


#%%
data['value'] = list(map(unify, data.unit, data.lower_value))
ld50 = data.groupby(['CasRN'])['value'].mean().reset_index()

tqdm.pandas()
ld50['SMILES'] = ld50.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
ld50.SMILES.isna().sum()
ld50 = ld50[ld50.SMILES.notna()].reset_index(drop = True)

ld50['category'] = pd.cut(ld50.value, bins = [0, 5, 50, 300, 2000, np.infty], labels = range(5))

ld50.to_excel('../oral.xlsx', header = True, index = False)
