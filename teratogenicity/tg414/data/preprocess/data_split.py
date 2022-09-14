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
data = pd.read_excel('tg414_noael.xlsx')


len(data['CasRN'].unique())

data['unit'].unique()
data['unit'].isna().sum()
data = data[data['unit'].notna()]
data = data[data['lower_value'].notna()]

casrn_na_idx = data[data['CasRN'] == '-'].index

data = data.drop(casrn_na_idx).reset_index(drop = True)


#%%
def unify(unit, value):
    if unit == 'ppm':
        v_ = value
    
    elif unit == 'mg/kg':
        v_ = value
    
    elif unit == 'mg/m^3':
        v_ = value * 0.001
    
    elif unit == 'ml/kg':
        v_ = value * 1000
    
    return v_

 
#%%
noael = data.copy()
noael = pd.concat([noael, pd.get_dummies(noael['admin type'])], axis = 1)
noael['value'] = list(map(unify, noael.unit, noael.lower_value))
noael = noael.groupby(['CasRN'])['value', 'all', 'dermal', 'inhalation', 'oral'].mean().reset_index()

# tqdm.pandas()
# noael['SMILES'] = noael.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
from urllib.request import urlopen

s_ = []
for i in tqdm(noael.CasRN):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + i + '/smiles'
        smiles = urlopen(url).read().decode('utf8')
        s_.append(smiles)
    except:
        s_.append('-')


noael['SMILES'] = s_
smiles_drop_idx = noael[noael.SMILES == '-'].index
noael = noael.drop(smiles_drop_idx, axis = 0).reset_index(drop = True)

noael.to_excel('../noael.xlsx', header = True, index = False)
