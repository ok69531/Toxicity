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
    data generate
    - group by CasRN
    - generate SMILES
'''

data = pd.read_excel('tg411_noael.xlsx')


len(data['CasRN'].unique())

data['unit'].unique()
data['unit'].isna().sum()
data = data[data['unit'].notna()]
data = data[data['lower_value'].notna()]

casrn_na_idx = data[data['CasRN'] == '-'].index

data = data.drop(casrn_na_idx).reset_index(drop = True)


#%%
def unify(unit, value):
    if unit == 'mg/kg':
        v_ = value
    
    elif unit == 'ml':
        v_ = value * 1000
    
    return v_


#%%
# gas data
noael = data.copy()
noael['value'] = list(map(unify, noael.unit, noael.lower_value))
noael = noael.groupby(['CasRN'])['value'].mean().reset_index()

tqdm.pandas()
noael['SMILES'] = noael.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
noael.SMILES.isna().sum()
noael = noael[noael['SMILES'].notna()].reset_index(drop = True)

noael['category'] = pd.cut(noael.value, bins = [0, 20, 200, np.infty], labels = range(3))

noael.to_excel('../noael.xlsx', header = True, index = False)
