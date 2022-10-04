#%%
import re
import cirpy
import openpyxl
import warnings

import pandas as pd
import numpy as np

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings("ignore")


#%%
df_tmp = pd.read_excel('tg421_raw.xlsx', header = 0)

df_tmp['Dose descriptor'].unique()
df_tmp['Dose descriptor'].value_counts()


endpoint = ['NOEL', 'NOAEL', 'NOAEC', 'NOEC']
noael = df_tmp[df_tmp['Dose descriptor'].map(lambda x: any(e in str(x) for e in endpoint))].reset_index(drop = True)


# Value에 other: ~~ 이렇게 돼있는거 제거
other_idx = [i for i in range(len(noael)) if 'other' in str(noael['Effect level'][i])]
noael['Effect level'][other_idx] = [re.sub(' other: ', '', noael['Effect level'][i]) for i in other_idx]


# Value에 ca. ~ 이렇게 돼있는거에서 ca. 제거
ca_idx = [i for i in range(len(noael)) if 'ca.' in str(noael['Effect level'][i])]
noael['Effect level'][ca_idx] = [re.sub('ca\. ', '', noael['Effect level'][i]) for i in ca_idx]


# 단위 통일
def unit_transform(string):
    if 'mg/m³' in string or 'mg/m3' in string:
        unit_ = re.sub('mg/m³|mg/m3', 'mg/m^3', string)
    
    elif 'g/m3' in string:
        unit_ = re.sub('g/m3', 'g/m^3', string)
    
    elif 'mg/l' in string:
        unit_ = re.sub('mg/l', 'mg/L', string)
    
    else:
        unit_ = string
    
    return unit_

noael['Effect level'] = noael['Effect level'].map(lambda x: unit_transform(str(x)).strip())


# Effect level에서 단위 추출
def extract_unit(string):
    try:
        u_ = re.findall('mg/m\^3|[a-zA-Z]*/[a-zA-Z]*|ppm', string)
        return u_[0]
    except:
        pass

tqdm.pandas()
noael['unit'] = noael['Effect level'].progress_apply(lambda x: extract_unit(str(x)))
noael.unit.unique()


# Value (value) 에서 괄호 안에 값 제거
np.unique([re.findall('\(.*?\)', str(i)) for i in noael['Effect level']])
noael['Effect level'] = [re.sub('\(.*?\)', '', str(i)) for i in noael['Effect level']]


# Value datframe
val_df = noael.copy()
val_df['Value_split'] = [val_df['Effect level'][i].split(' ') for i in range(len(val_df))]


# range Value
range_idx = [i for i in range(len(val_df)) if '-' in str(noael['Effect level'][i])]
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
val_df['lower_ineq'] = ''

for i in tqdm(idx):
    if isFloat(val_df['Value_split'][i][0]):
        val_df['lower_ineq'][i] = np.nan
        
    else:
        val_df['lower_ineq'][i] = val_df['Value_split'][i][0]
        val_df['Value_split'][i].remove(val_df['Value_split'][i][0])


# 값
val_df['lower_value'] = ''

for i in tqdm(idx):
    if isFloat(''.join(val_df['Value_split'][i][:4])):
        val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:4]))
        del val_df['Value_split'][i][:4]
        
    elif isFloat(''.join(val_df['Value_split'][i][:3])):
        val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:3]))
        del val_df['Value_split'][i][:3]
    
    elif isFloat(''.join(val_df['Value_split'][i][:2])):
        val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:2]))
        del val_df['Value_split'][i][:2]
    
    elif isFloat(''.join(val_df['Value_split'][i][:1])):
        val_df['lower_value'][i] = float(''.join(val_df['Value_split'][i][:1]))
        del val_df['Value_split'][i][:1]


# 단위
for i in idx:
    try:
        val_df['Value_split'][i].remove(val_df['unit'][i])
    except ValueError:
        pass


#%%
# range_idx

# 부등호
val_df['upper_ineq'] = ''
       
for i in range_idx:
    try:
        range_ineq = re.findall('>\=|<\=|>|<', val_df['Effect level'][i])
        val_df['lower_ineq'][i] = range_ineq[0]
        val_df['upper_ineq'][i] = range_ineq[1]
        
        val_df['Value_split'][i] = [j for j in val_df['Value_split'][i] if j not in range_ineq]
        
    except IndexError:
        val_df['lower_ineq'][i] = np.nan
        val_df['upper_ineq'][i] = np.nan


# value
val_df['upper_value'] = ''

for i in range_idx:
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


#%%
def check_nan(string):
    return string == string
        

def admin_type(string):
    if check_nan(string):
        string = string.lower()
        
        if 'inhalation' in string:
            type_ = 'inhalation'
        
        elif 'oral' in string:
            type_ = 'oral'
        
        elif 'dermal' in string:
            type_ = 'dermal'
        
        else:
            type_ = np.nan
    
    else:
        type_ = np.nan
        
    return type_



val_df['admin type'] = val_df['Route of administration'].map(lambda x: admin_type(str(x)))
val_df['admin type'].value_counts()


#%%
val_df.unit.unique()
# val_df.unit[val_df.unit == 'ppmppm'] = 'ppm'

val_df.to_excel('tg421.xlsx', header = True, index = False)
