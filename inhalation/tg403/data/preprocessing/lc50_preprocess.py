#%%
import re
import openpyxl

import pandas as pd
import numpy as np 

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)


#%%
lc50_tmp = pd.read_excel('tg403_raw.xlsx', header = 0)

lc50_tmp['Dose descriptor'].unique()
lc50_idx = [i for i in range(len(lc50_tmp)) if 'LC50' in str(lc50_tmp['Dose descriptor'][i])]
lc50_tmp['Dose descriptor'][lc50_idx].unique()


lc50 = lc50_tmp[(lc50_tmp['Dose descriptor'] == 'LC50') | 
                (lc50_tmp['Dose descriptor'] == 'other: approximate LC50') | 
                (lc50_tmp['Dose descriptor'] == 'other: LC50')]
lc50.reset_index(drop = True, inplace = True)
lc50['Dose descriptor'].unique()


# Value에 other: ~~ 이렇게 돼있는거 제거
other_idx = [i for i in range(len(lc50)) if 'other' in str(lc50['Effect level'][i])]
lc50['Effect level'][other_idx] = [re.sub(' other: ', '', lc50['Effect level'][i]) for i in other_idx]


# Value에 ca. ~ 이렇게 돼있는거에서 ca. 제거
ca_idx = [i for i in range(len(lc50)) if 'ca.' in str(lc50['Effect level'][i])]
lc50['Effect level'][ca_idx] = [re.sub('ca\. ', '', lc50['Effect level'][i]) for i in ca_idx]


# 단위 통일
def unit_transform(string):
    if 'mg/m³'  in string or 'mg/m3' in string:
        unit_ = re.sub('mg/m³|mg/m3', 'mg/m^3', string)
    
    elif 'g/m3' in string:
        unit_ = re.sub('g/m3', 'g/m^3', string)
    
    elif 'mg/l' in string:
        unit_ = re.sub('mg/l', 'mg/L', string)
    
    else:
        unit_ = string
    
    return unit_


lc50['Effect level'] = lc50['Effect level'].map(lambda x: unit_transform(str(x)))


# Value (value) 에서 괄호 안에 값 제거
np.unique([re.findall('\(.*?\)', str(i)) for i in lc50['Effect level']])
lc50['Effect level'] = [re.sub('\(.*?\)', '', str(i)) for i in lc50['Effect level']]


# Value datframe
val_df = lc50.copy()
val_df['Value_split'] = [val_df['Effect level'][i].split(' ') for i in range(len(val_df))]


# range Value
range_idx = [i for i in range(len(val_df)) if '-' in str(lc50['Effect level'][i])]
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
# val_df['unit'] = ''
u_ = ['µg/m^3', 'mg/m^3', 'g/m^3', 'mg/L', 'ppm']
val_df['unit'] = val_df['Effect level'].apply(lambda x: ''.join(y for y in x.split() if y in u_))

for i in idx:
    try:
        val_df['Value_split'][i].remove(val_df['unit'][i])
    except ValueError:
        pass


# time
val_df['Exp. duration'].unique()

val_df['time'] = ''
val_df['time'][val_df['Exp. duration'].isna()] = np.nan

time_idx = val_df[val_df['Exp. duration'].isna() == False].index

for i in time_idx:
    if isFloat(val_df['Exp. duration'][i]):
        val_df['time'][i] = np.nan
    
    elif val_df['Exp. duration'][i].split(' ')[-1] == 'h':
        val_df['time'][i] = float(val_df['Exp. duration'][i].split(' ')[0])
    
    elif val_df['Exp. duration'][i].split(' ')[-1] == 'min':
        val_df['time'][i] = float(val_df['Exp. duration'][i].split(' ')[0])/60


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
# SMILES = 0 인 value 변환
from urllib.request import urlopen
from urllib.parse import quote 

def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return '-'

# index of SMILES = 0 or SMILES = NaN
smi_idx = list(val_df['SMILES'][val_df['SMILES'] == 0].index) + list(val_df[val_df.SMILES.isna()].index)

smi_ =  [CIRconvert(i) for i in tqdm(val_df['CasRN'][smi_idx])]
val_df['SMILES'][smi_idx] = smi_


#%%
val_df.drop(['Value_split'], axis = 1, inplace = True)
val_df.to_excel('tg403_lc50.xlsx', header = True, index = False)
