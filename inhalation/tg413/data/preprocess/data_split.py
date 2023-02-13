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

data = pd.read_excel('tg413_noael.xlsx')


len(data['CasRN'].unique())

data['unit'].unique()
data['unit'].isna().sum()
data = data[data['unit'].notna()]
data = data[data['lower_value'].notna()]

casrn_na_idx = data[data['CasRN'] == '-'].index

data = data.drop(casrn_na_idx).reset_index(drop = True)


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
noael_gas_tmp = data[data['inhale type'] == 'gas']
noael_gas_tmp['value'] = list(map(unify, noael_gas_tmp.unit, noael_gas_tmp.lower_value))
noael_gas = noael_gas_tmp.groupby(['CasRN'])['value'].mean().reset_index()

tqdm.pandas()
noael_gas['SMILES'] = noael_gas.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
noael_gas.SMILES.isna().sum()
noael_gas = noael_gas[noael_gas['SMILES'].notna()].reset_index(drop = True)

noael_gas['category'] = pd.cut(noael_gas.value, bins = [0, 50, 250, np.infty], labels = range(3))

noael_gas.to_excel('../gas.xlsx', header = True, index = False)


#%%
# vapour data
noael_vap_tmp = data[data['inhale type'] == 'vapour']
noael_vap_tmp['value'] = list(map(unify, noael_vap_tmp.unit, noael_vap_tmp.lower_value))
noael_vap = noael_vap_tmp.groupby(['CasRN'])['value'].mean().reset_index()

noael_vap['SMILES'] = noael_vap.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
noael_vap.SMILES.isna().sum()
noael_vap = noael_vap[noael_vap['SMILES'].notna()].reset_index(drop = True)

noael_vap['category'] = pd.cut(noael_vap.value, bins =[0, 0.2, 1.0, np.infty], labels = range(3))
noael_vap.to_excel('../vapour.xlsx', header = True, index = False)


#%%
# vapour data
noael_aer_tmp = data[data['inhale type'] == 'aerosol']
noael_aer_tmp['value'] = list(map(unify, noael_aer_tmp.unit, noael_aer_tmp.lower_value))
noael_aer = noael_aer_tmp.groupby(['CasRN'])['value'].mean().reset_index()
# noael_aer = noael_vap_tmp.groupby(['CasRN', 'SMILES'])['time', 'value'].mean().reset_index()

noael_aer['SMILES'] = noael_aer.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
noael_aer.SMILES.isna().sum()
noael_aer = noael_aer[noael_aer['SMILES'].notna()].reset_index(drop = True)

noael_aer['category'] = pd.cut(noael_aer.value, bins =[0, 0.02, 0.2, np.infty], labels = range(3))

noael_aer.to_excel('../aerosol.xlsx', header = True, index = False)
