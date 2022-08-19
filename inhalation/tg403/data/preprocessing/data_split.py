#%%
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

data = pd.read_excel('tg403_lc50.xlsx')


len(data['CasRN'].unique())

data['unit'].unique()
data['unit'].isna().sum()
data = data[data['unit'].notna()]
data = data[data['lower_value'].notna()]
data = data[data['SMILES'].notna()]

casrn_na_idx = data[data['CasRN'] == '-'].index
smiles_na_idx = data[data['SMILES'] == '-'].index

data = data.drop(list(casrn_na_idx) + list(smiles_na_idx)).reset_index(drop = True)


#%%
def unify(unit, value):
    if unit == 'mg/L':
        v_ = value
    
    elif unit == 'ppm':
        v_ = value
    
    elif unit == 'g/m^3':
        v_ = value
    
    elif unit == 'mg/m^3':
        v_ = value * 0.001
    
    elif unit == 'µg/m^3':
        v_ = value * 0.000001
    
    return v_


#%%
# gas data
lc50_gas_tmp = data[data['inhale type'] == 'gas']
lc50_gas_tmp['value'] = list(map(unify, lc50_gas_tmp.unit, lc50_gas_tmp.lower_value))
lc50_gas = lc50_gas_tmp.groupby(['CasRN', 'SMILES'])['time','value'].mean().reset_index()
lc50_gas['category'] = pd.cut(lc50_gas.value, bins = [0, 100, 500, 2500, 20000, np.infty], labels = range(5))

lc50_gas.to_excel('../gas.xlsx', header = True, index = False)


#%%
# vapour data
lc50_vap_tmp = data[data['inhale type'] == 'vapour']
lc50_vap_tmp['value'] = list(map(unify, lc50_vap_tmp.unit, lc50_vap_tmp.lower_value))
lc50_vap = lc50_vap_tmp.groupby(['CasRN', 'SMILES'])['time', 'value'].mean().reset_index()
lc50_vap['category'] = pd.cut(lc50_vap.value, bins =[0, 0.5, 2.0, 10, 20, np.infty], labels = range(5))

lc50_vap.to_excel('../vapour.xlsx', header = True, index = False)


#%%
# vapour data
lc50_aer_tmp = data[data['inhale type'] == 'aerosol']
lc50_aer_tmp['value'] = list(map(unify, lc50_aer_tmp.unit, lc50_aer_tmp.lower_value))
lc50_aer = lc50_vap_tmp.groupby(['CasRN', 'SMILES'])['time', 'value'].mean().reset_index()
lc50_aer['category'] = pd.cut(lc50_vap.value, bins =[0, 0.05, 0.5, 1.0, 5.0, np.infty], labels = range(5))

lc50_aer.to_excel('../aerosol.xlsx', header = True, index = False)
