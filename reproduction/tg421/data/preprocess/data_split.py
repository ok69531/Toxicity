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

data = pd.read_excel('tg421.xlsx')


len(data['CasRN'].unique())

data['unit'].unique()
data['unit'].isna().sum()
data = data[data['unit'].notna()]
data = data[data['lower_value'].notna()]

casrn_na_idx = data[data['CasRN'] == '-'].index
drop_na_idx = data[data.unit == 'eq/kg'].index

data = data.drop(casrn_na_idx.tolist() + drop_na_idx.tolist()).reset_index(drop = True)

#%%
# mg/kg = mg/l = ppm
def unify(unit, value):
    if unit == 'mg/L':
        v_ = value
    
    elif unit == 'ppm':
        v_ = value
    
    elif unit == 'mg/kg':
        v_ = value
    
    elif unit == 'mg/m^3':
        v_ = value * 0.001
    
    elif unit == 'µg/m^3':
        v_ = value * 0.000001
    
    return v_


#%%
# oral data
oral_tmp = data[data['admin type'] == 'oral']
oral_tmp['value'] = list(map(unify, oral_tmp.unit, oral_tmp.lower_value))
oral = oral_tmp.groupby(['CasRN'])['value'].mean().reset_index()

tqdm.pandas()
oral['SMILES'] = oral.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
oral.SMILES.isna().sum()
oral = oral[oral['SMILES'].notna()].reset_index(drop = True)

oral.to_excel('../oral.xlsx', header = True, index = False)


#%%
# inhalation data
inhale_tmp = data[data['admin type'] == 'inhalation']
inhale_tmp['value'] = list(map(unify, inhale_tmp.unit, inhale_tmp.lower_value))
inhale = inhale_tmp.groupby(['CasRN'])['value'].mean().reset_index()

tqdm.pandas()
inhale['SMILES'] = inhale.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
inhale.SMILES.isna().sum()
inhale = inhale[inhale['SMILES'].notna()].reset_index(drop = True)

inhale.to_excel('../inhale.xlsx', header = True, index = False)


#%%
# dermal data
dermal_tmp = data[data['admin type'] == 'dermal']
dermal_tmp['value'] = list(map(unify, dermal_tmp.unit, dermal_tmp.lower_value))
dermal = dermal_tmp.groupby(['CasRN'])['value'].mean().reset_index()

tqdm.pandas()
dermal['SMILES'] = dermal.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
dermal.SMILES.isna().sum()
dermal = dermal[dermal['SMILES'].notna()].reset_index(drop = True)

dermal.to_excel('../dermal.xlsx', header = True, index = False)
