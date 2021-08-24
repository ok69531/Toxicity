#%%
import pymysql
import os
import re

import pandas as pd
import numpy as np 

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)
#%%
oral_tmp = pd.read_excel('C:/Users/SOYOUNG/Desktop/toxic/data/oecd_echemportal/Data tmp/oral420_423_425.xlsx', header = 0, sheet_name = 'Sheet1')

oral_tmp.shape
len(oral_tmp['CasRN'].unique())

oral_tmp['Descriptor'].value_counts()
(oral_tmp['CasRN'] == '-').sum()

#%%
# descriptor = pd.DataFrame({'descriptor': oral_tmp['Descriptor'].unique()})
# descriptor['is_ld50'] = [len(re.findall('LD50', str(descriptor['descriptor'][i]))) for i in range(descriptor.shape[0])]

oral = oral_tmp[(oral_tmp['Descriptor'] == 'LD50') | (oral_tmp['Descriptor'] == 'approximate LD50')]
oral.reset_index(drop = True, inplace = True)
oral['Descriptor'].unique()

# Value에 other: ~~ 이렇게 돼있는거 제거
other = pd.DataFrame([len(re.findall('other', oral['Value'][i])) for i in range(oral.shape[0])])
other_idx = other[other[0] == 1].index
oral.iloc[other_idx]
oral.drop(other_idx, axis = 0, inplace = True)
oral.reset_index(drop = True, inplace = True)

# Value 이상한거 제거 ,, 
etc = pd.DataFrame([len(re.findall('effects|guideline|off|h|d|step|determinable', oral['Value'][i])) for i in range(oral.shape[0])])
etc_idx = etc[etc[0] == 1].index
oral.iloc[etc_idx]
oral = oral.drop(etc_idx, axis = 0).reset_index(drop = True)

# Value에 ca. ~ 이렇게 돼있는거에서 ca. 제거
ca = pd.DataFrame([len(re.findall('ca\. ', oral['Value'][i])) for i in range(oral.shape[0])])
ca_idx = ca[ca[0] != 0].index
oral.iloc[ca_idx]
for i in ca_idx:
    oral['Value'][i] = re.sub('ca\. ', '', oral['Value'][i])


# Value datframe
val_df = pd.DataFrame(columns = {'Chemical_Name', 'CasRN', 'Descriptor', 'Value_tmp', 'Value_split', 'lower_ineq', 
                                 'lower_value', 'upper_ineq', 'upper_value', 'unit', 'bw'})
val_df.columns = ['Chemical_Name'] + ['CasRN'] + ['Descriptor'] + ['Value_tmp'] + ['Value_split'] + ['lower_ineq'] + ['lower_value'] + ['upper_ineq'] + ['upper_value'] + ['unit'] + ['bw']
val_df['Chemical_Name'] = oral['Chemical_Name']
val_df['Descriptor'] = oral['Descriptor']
val_df['CasRN'] = oral['CasRN']
val_df['Value_tmp'] = oral['Value']

# for i in range(val_df.shape[0]):
#     val_df['Value_tmp'][i] = re.sub('mg/m³', 'mg/m^3', val_df['Value_tmp'][i])


val_df['Value_split'] = [val_df['Value_tmp'][i].split(' ') for i in range(len(val_df))]


# range Value
val_df['range_cat'] = [len(re.findall('-', val_df['Value_tmp'][i])) for i in range(val_df.shape[0])]

range_idx = val_df['range_cat'][val_df['range_cat'] != 0].index
idx = set(list(range(val_df.shape[0]))) - set(range_idx)


#%%
# range가 아닌 value들부터

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


# 부등호
for i in tqdm(idx):
    if isFloat(val_df['Value_split'][i][0]):
        val_df['lower_ineq'][i] = '='
    else:
        val_df['lower_ineq'][i] = val_df['Value_split'][i][0]
        val_df['Value_split'][i].remove(val_df['Value_split'][i][0])


# 값
for i in tqdm(idx):
    if isFloat(''.join(val_df['Value_split'][i][:3])):
        val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:3]))
        del val_df['Value_split'][i][:3]
    
    elif isFloat(''.join(val_df['Value_split'][i][:2])):
        val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:2]))
        del val_df['Value_split'][i][:2]
    
    elif isFloat(''.join(val_df['Value_split'][i][:1])):
        val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:1]))
        del val_df['Value_split'][i][:1]


# bw
for i in tqdm(idx):
    try:
        if val_df['Value_split'][i][-1] == 'bw':
            val_df['bw'][i] = val_df['Value_split'][i][-1]
            del val_df['Value_split'][i][-1]
        else:
            val_df['bw'][i] = np.nan
    except IndexError:
        val_df['bw'][i] = np.nan



# 단위
a = [val_df['Value_split'][i][-1] for i in idx if len(val_df['Value_split'][i]) != 0]
set(a)


for i in tqdm(idx):
    try:
        val_df['unit'][i] = re.findall('mg/kg|mL/kg', val_df['Value_tmp'][i])[0]
        val_df['Value_split'][i].remove(re.findall('mg/kg|mL/kg', val_df['Value_tmp'][i])[0])
    except IndexError:
        val_df['unit'][i] = np.nan






#%%
# range_idx

# 부등호
for i in tqdm(range_idx):
    try:
        range_ineq = re.findall('>\=|<\=|>|<', val_df['Value_tmp'][i])
        
        if len(range_ineq) == 1:
            if range_ineq[0] == val_df['Value_split'][i][0]:
                val_df['lower_ineq'][i] = range_ineq[0]
                val_df['Value_split'][i].remove(range_ineq[0])
            else:
                val_df['upper_ineq'][i] = range_ineq[0]
                val_df['Value_split'][i].remove(range_ineq[0])
        
        elif len(range_ineq) == 2:
            val_df['lower_ineq'][i] = range_ineq[0]
            val_df['upper_ineq'][i] = range_ineq[1]
            
            val_df['Value_split'][i] = [j for j in val_df['Value_split'][i] if j not in range_ineq]
        
    except IndexError:
        val_df['lower_ineq'][i] = np.nan
        val_df['upper_ineq'][i] = np.nan


# value
for i in tqdm(range_idx):
    try:
        if val_df['Value_split'][i].index('-') == 1:
            val_df['lower_value'][i] = float(val_df['Value_split'][i][0])
        # val_df['Value_split'][i].remove(val_df['Value_split'][i][0])
        
            if isFloat(''.join(val_df['Value_split'][i][2:4])):
                val_df['upper_value'][i] = float(''.join(val_df['Value_split'][i][2:4]))
                # val_df['Value_split'][i].remove(''.join(val_df['Value_split'][i][2:4]))
            else:
                val_df['upper_value'][i] = float(val_df['Value_split'][i][2])
                # val_df['Value_split'][i].remove(val_df['Value_split'][i][2])
            
        elif val_df['Value_split'][i].index('-') == 2:
            val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:2]))
            val_df['upper_value'][i] = float(''.join(val_df['Value_split'][i][3:5]))
    
    except ValueError:
        val_df['lower_value'][i] = np.nan
        val_df['upper_value'][i] = np.nan
        


# unit
for i in tqdm(range_idx):
    try:
        val_df['unit'][i] = re.findall('mg/kg|mL/kg', val_df['Value_tmp'][i])[0]
        val_df['Value_split'][i] = [j for j in val_df['Value_split'][i] if j not in val_df['unit'][i]]
    except IndexError:
        val_df['unit'][i] = np.nan


# bw
for i in tqdm(range_idx):
    try:
        val_df['bw'][i] = re.findall('bw', val_df['Value_tmp'][i])[0]
        val_df['Value_split'][i] = [j for j in val_df['Value_split'][i] if j not in val_df['bw'][i]]
    except IndexError:
        val_df['bw'][i] = np.nan




#%%
val_df.iloc[range_idx]


val_df.drop(['Value_split', 'range_cat'], axis = 1, inplace = True)
val_df.to_excel('oral420_423_425_ld50.xlsx', header = True, index = False)