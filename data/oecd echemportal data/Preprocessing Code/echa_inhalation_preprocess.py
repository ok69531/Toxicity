#%%
''' 
    1. LC50 데이터 전처리
        > 부등호, 값, 단위, air, nominal/analytical 구분    >>     lower bound와 upper bound로 나눠야 하나
        > other이 포함된 행 제거
        > 단위 통일
        > 여러개 있는 값은 어떤 것을 쓸 건지 결정 ,, 보수적으로? 

    2. Chemical-Disease에서 Slimapping을 이용해 호흡기, 신경계 구분 
    
    3. 1, 2에서 만든 데이터 합치기
    
    
    목표 : LD50 값을 예측하는 회귀모형 혹은 
           LD50 값의 일정 Threshold를 기준으로 한 classification model의 적합 

        1. 기존에 호흡기 데이터에 LC값 추가해서 chemical의 독성 예측
        2. LD50 값을 예측하는 회귀모형
        3. LD50 값의 일정 Threshold를 기준으로 한 classification model의 적합
'''

#%%
import pymysql
import os
import re

import pandas as pd
import numpy as np 

pd.set_option('mode.chained_assignment', None)
#%%
lc50_tmp = pd.read_excel('C:/Users/SOYOUNG/Desktop/toxic/preprocessed_data/echa_inhalation.xlsx', header = 0)


lc50 = lc50_tmp[(lc50_tmp['Descriptor'] == 'LC50') | 
                (lc50_tmp['Descriptor'] == 'other: approximate LC50') | 
                (lc50_tmp['Descriptor'] == 'other: LC50')|
                (lc50_tmp['Descriptor'] == 'other: LD50')]
lc50.reset_index(drop = True, inplace = True)
lc50['Descriptor'].unique()


# Value에 other: ~~ 이렇게 돼있는거 제거
other = pd.DataFrame([len(re.findall('other', lc50['Value'][i])) for i in range(lc50.shape[0])])
other_idx = other[other[0] == 1].index
lc50.drop(other_idx, axis = 0, inplace = True)
lc50.reset_index(drop = True, inplace = True)


# Value에 ca. ~ 이렇게 돼있는거에서 ca. 제거
ca = pd.DataFrame([len(re.findall('ca\. ', lc50['Value'][i])) for i in range(lc50.shape[0])])
ca_idx = ca[ca[0] != 0].index
for i in ca_idx:
    lc50['Value'][i] = re.sub('ca\. ', '', lc50['Value'][i])


# Value datframe
val_df = pd.DataFrame(columns = {'Chemical_Name', 'Descriptor', 'CasRN', 'Value_tmp', 'Value_split', 'lower_ineq', 
                                 'lower_value', 'upper_ineq', 'upper_value', 'unit', 'air', 'nominal/analytical'})
val_df.columns = ['Chemical_Name'] + ['Descriptor'] + ['CasRN'] + ['Value_tmp'] + ['Value_split'] + ['lower_ineq'] + ['lower_value'] + ['upper_ineq'] + ['upper_value'] + ['unit'] + ['air'] + ['nominal/analytical']
val_df['Chemical_Name'] = lc50['Chemical_Name']
val_df['Descriptor'] = lc50['Descriptor']
val_df['CasRN'] = lc50['CasRN']
val_df['Value_tmp'] = lc50['Value']

for i in range(val_df.shape[0]):
    val_df['Value_tmp'][i] = re.sub('mg/m³', 'mg/m^3', val_df['Value_tmp'][i])


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
for i in idx:
    if isFloat(val_df['Value_split'][i][0]):
        val_df['lower_ineq'][i] = '='
    else:
        val_df['lower_ineq'][i] = val_df['Value_split'][i][0]
        val_df['Value_split'][i].remove(val_df['Value_split'][i][0])


# 값
for i in idx:
    if isFloat(''.join(val_df['Value_split'][i][:3])):
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
        val_df['unit'][i] = re.findall('mg/m\^3|mg/L|ppm', val_df['Value_tmp'][i])[0]
        val_df['Value_split'][i].remove(re.findall('mg/m\^3|mg/L|ppm', val_df['Value_tmp'][i])[0])
    except IndexError:
        val_df['unit'][i] = np.nan


# air 
for i in idx:
    try:
        if val_df['Value_split'][i][0] == 'air':
            val_df['air'][i] = val_df['Value_split'][i][0]
            del val_df['Value_split'][i][0]
        else:
            val_df['air'][i] = np.nan
    except IndexError:
        val_df['air'][i] = np.nan


# nominal / analytical
for i in idx:
    try:
        val_df['nominal/analytical'][i] = re.sub('\(|\)', '', val_df['Value_split'][i][-1])
        del val_df['Value_split'][i][-1]
    except IndexError:
        val_df['nominal/analytical'][i] = np.nan



#%%
# range_idx

# 부등호
for i in range_idx:
    try:
        range_ineq = re.findall('>\=|<\=|>|<', val_df['Value_tmp'][i])
        val_df['lower_ineq'][i] = range_ineq[0]
        val_df['upper_ineq'][i] = range_ineq[1]
        
        val_df['Value_split'][i] = [j for j in val_df['Value_split'][i] if j not in range_ineq]
        
    except IndexError:
        val_df['lower_ineq'][i] = np.nan
        val_df['upper_ineq'][i] = np.nan


# value
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
        


# unit
for i in range_idx:
    try:
        val_df['unit'][i] = re.findall('mg/m\^3|mg/L|ppm', val_df['Value_tmp'][i])[0]
        val_df['Value_split'][i] = [j for j in val_df['Value_split'][i] if j not in val_df['unit'][i]]
    except IndexError:
        val_df['unit'][i] = np.nan


# air
for i in range_idx:
    try:
        val_df['air'][i] = re.findall('air', val_df['Value_tmp'][i])[0]
        val_df['Value_split'][i] = [j for j in val_df['Value_split'][i] if j not in val_df['air'][i]]
    except IndexError:
        val_df['air'][i] = np.nan


# nominal/analytical
for i in range_idx:
    tmp = re.findall('\([a-z]*\)', val_df['Value_tmp'][i])
    try:
        val_df['nominal/analytical'][i] = re.sub('\(|\)', '', tmp[0])
    except IndexError:
        val_df['nominal/analytical'][i] = np.nan


#%%
val_df.iloc[range_idx]
val_df.iloc[4871]


val_df.drop(['Value_split', 'range_cat'], axis = 1, inplace = True)
val_df.to_excel('inhalation403_lc50.xlsx', header = True, index = False)
