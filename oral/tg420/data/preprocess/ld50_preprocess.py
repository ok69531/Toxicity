#%%
import re
import openpyxl

import pandas as pd
import numpy as np 

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)


#%%
df_tmp = pd.read_excel('tg420_raw.xlsx', header = 0)

df_tmp['Dose descriptor'].unique()
df_tmp['Dose descriptor'].value_counts()

ld50_descriptor = ['LD50', 
                   'LD50 cut-off', 
                   'other: LD50 cut-off', 
                   'other: LD50 cut-off (rat)', 
                   'other: LD50 cut-off value', 
                   'other: LD50 (cut off)']
ld50 = df_tmp[df_tmp['Dose descriptor'].isin(ld50_descriptor)].reset_index(drop = True)
ld50['Dose descriptor'].unique()


# Value에 other: ~~ 이렇게 돼있는거 제거
other_idx = [i for i in range(len(ld50)) if 'other' in str(ld50['Effect level'][i])]
ld50['Effect level'][other_idx] = [re.sub(' other: ', '', ld50['Effect level'][i]) for i in other_idx]


# Value에 ca. ~ 이렇게 돼있는거에서 ca. 제거
ca_idx = [i for i in range(len(ld50)) if 'ca.' in str(ld50['Effect level'][i])]
ld50['Effect level'][ca_idx] = [re.sub('ca\. ', '', ld50['Effect level'][i]) for i in ca_idx]


# Effect level에서 단위 추출
def extract_unit(string):
    try:
        u_ = re.findall('[a-zA-Z]*/[a-zA-Z]*', string)
        return u_[0]
    except:
        pass

tqdm.pandas()
ld50['unit'] = ld50['Effect level'].progress_apply(lambda x: extract_unit(str(x)))
ld50.unit.unique()


# 단위가 mg/kg, g/kg인 데이터만 추출
ld50 = ld50[ld50.unit.isin(['mg/kg', 'g/kg'])].reset_index(drop = True)


# Value (value) 에서 괄호 안에 값 제거
np.unique([re.findall('\(.*?\)', str(i)) for i in ld50['Effect level']])
ld50['Effect level'] = [re.sub('\(.*?\)', '', str(i)) for i in ld50['Effect level']]


# Value datframe
val_df = ld50.copy()
val_df['Value_split'] = [val_df['Effect level'][i].split(' ') for i in range(len(val_df))]


# range Value
range_idx = [i for i in range(len(val_df)) if '-' in str(ld50['Effect level'][i])]
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
val_df.unit.unique()
val_df.drop(['Value_split'], axis = 1, inplace = True)

val_df.to_excel('tg420_ld50.xlsx', header = True, index = False)
