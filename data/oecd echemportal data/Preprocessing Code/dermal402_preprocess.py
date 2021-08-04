#%%
import pymysql
import os
import re

import pandas as pd
import numpy as np 

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)

#%%
dermal402 = pd.read_excel('C:/Users/SOYOUNG/Desktop/toxic/data/oecd_echemportal/Data tmp/dermal402.xlsx', header = 0)

dermal402['Descriptor'].value_counts()
len(dermal402['CasRN'].unique())
(dermal402['CasRN'] == 'unknown').sum()


#%%
ld50_tmp = dermal402[(dermal402['Descriptor'] == 'LD50') | 
                     (dermal402['Descriptor'] == 'approximate LD50') | 
                     (dermal402['Descriptor'] == 'other: LD50 dermal')]
ld50_tmp.reset_index(drop = True, inplace = True)
ld50_tmp['Descriptor'].unique()


# Value에 ca. ~ 이렇게 돼있는거에서 ca. 제거
ca = pd.DataFrame([len(re.findall('ca\. ', ld50_tmp['Value'][i])) for i in range(ld50_tmp.shape[0])])
ca_idx = ca[ca[0] != 0].index
ld50_tmp.iloc[ca_idx]
for i in ca_idx:
    ld50_tmp['Value'][i] = re.sub('ca\. ', '', ld50_tmp['Value'][i])


# Value에 test mat, not determinable due to absence of adverse toxic effects 제거
drop_idx = ld50_tmp[(ld50_tmp['Value'] == 'test mat.') | (ld50_tmp['Value'] == 'not determinable due to absence of adverse toxic effects')].index
ld50_tmp.drop(drop_idx, axis = 0, inplace = True)
ld50_tmp.reset_index(inplace = True, drop = True)

# Value에 other: ~~ 이렇게 돼있는거 제거
other = pd.DataFrame([len(re.findall('other', ld50_tmp['Value'][i])) for i in range(ld50_tmp.shape[0])])
other_idx = other[other[0] == 1].index
ld50_tmp.iloc[other_idx]

ld50 = ld50_tmp.drop(other_idx, axis = 0)
ld50.reset_index(drop = True, inplace = True)



# Value datframe
val_df = pd.DataFrame(columns = {'Chemical_Name', 'Descriptor', 'CasRN', 'Value_tmp', 'Value_split', 'lower_ineq', 
                                 'lower_value', 'upper_ineq', 'upper_value', 'unit', 'bw'})
val_df.columns = ['Chemical_Name'] + ['Descriptor'] + ['CasRN'] + ['Value_tmp'] + ['Value_split'] + ['lower_ineq'] + ['lower_value'] + ['upper_ineq'] + ['upper_value'] + ['unit'] + ['bw']
val_df['Chemical_Name'] = ld50['Chemical_Name']
val_df['Descriptor'] = ld50['Descriptor']
val_df['CasRN'] = ld50['CasRN']
val_df['Value_tmp'] = ld50['Value']

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
        val_df['bw'][i] = re.findall('bw', val_df['Value_tmp'][i])[0]
        val_df['Value_split'][i].remove(re.findall('bw', val_df['Value_tmp'][i])[0])
    except IndexError:
        val_df['bw'][i] = np.nan
        


# 단위
a = [val_df['Value_split'][i][-1] for i in idx if len(val_df['Value_split'][i]) != 0]
set(a)

for i in tqdm(idx):
    try:
        val_df['unit'][i] = re.findall('mL/kg|mg/kg', val_df['Value_tmp'][i])[0]
        val_df['Value_split'][i].remove(re.findall('mL/kg|mg/kg', val_df['Value_tmp'][i])[0])
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
        val_df['unit'][i] = re.findall('mL/kg|mg/kg', val_df['Value_tmp'][i])[0]
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
val_df.to_excel('dermal402_ld50.xlsx', header = True, index = False)
