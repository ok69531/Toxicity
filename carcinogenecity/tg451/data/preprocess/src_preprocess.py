#%%
import re
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup


#%%
with open('tg451_page_src.json', 'r') as file:
    df = pd.DataFrame(json.load(file))


#%%
def remove_bracket(string):
    clean_string = re.sub('<.*?>', '', str(string))
    return clean_string


#%%
result_ = []

for i in tqdm(range(len(df))):
    try: 
        soup = BeautifulSoup(df.src[i], 'html.parser')
        chem_dict = {}
        
        # chemical name
        chem_name = soup.find('div', attrs = {'id': 'SubstanceName'}).find_next('h1').text
        chem_dict['Chemical'] = chem_name


        # casrn
        casrn_tmp = soup.find('div', attrs = {'class': 'container'}).find_next('strong').text
        casrn = re.sub('\n|\t', '', casrn_tmp).split( )[-1]
        chem_dict['CasRN'] = casrn
        
        
        # route of administration
        try:
            admin_tmp = soup.find('h3', attrs = {'id': 'sMaterialsAndMethods'})
            
            admin_tmp = admin_tmp.find_next_siblings('dl')
            admin_idx = [remove_bracket(x.find_next('dt')) for x in admin_tmp].index('Route of administration:')
            
            route_tmp = admin_tmp[admin_idx]
            
            chem_dict['Route of administration'] = remove_bracket(route_tmp.find_next('dd'))
            
        except ValueError:
            chem_dict['Route of administration'] = np.nan


        # experiment results
        result_and_discussion = soup.find('h3', attrs={'id': 'sResultsAndDiscussion'})
        table_list = result_and_discussion.find_next_sibling('div').find_all('dl')


        for tab in table_list:
            chem_dict_ = chem_dict.copy()
            
            key = [re.sub(':', '', i.text).strip() for i in tab.find_all('dt')]
            value = [i.text.strip() for i in tab.find_all('dd')]
                                    
            if len(key) == len(value) and key[0] != '' and value[0] != 'Key result':
                result_dict = dict(zip(key, value))
                # result_dict = {key[i]: re.sub('<.*?>', '', cell.text).strip() for i, cell in enumerate(tab.find_all('dd'))}
            
            elif len(key) == len(value) and key[0] == '' and value[0] == 'Key result':
                result_dict = dict(zip(key[1:], value[1:]))
            
            elif len(key) != len(value) and key[0] == '' and value[0] == 'Key result':
                key = key[1:]
                value_ = value[1:len(key)] + ['. '.join(value[len(key):])]
                result_dict = dict(zip(key, value_))
            
            chem_dict_.update(result_dict)
            result_.append(chem_dict_)

    except AttributeError:
        pass
    


# %%
result = pd.DataFrame(result_)


#%%
'''
    1. nan이 아닌 경우
    aerosol -> 보수적인 쪽으로. Dusts and Mists로 넣으면 됨 (aerosol include mists, smokes, fumes, and dusts)
    
    vapour + aerosol 같이 여러개가 혼합된 경우 보수적인 카테고리로 포함
    
    aerosol, vapour, gas, dust가 포함되는 애들 찾고
    aerosol이 포함되면 aerosol category
    aerosol 없이 vapour or vapor or vaporization 포함되면 vapour
    aerosol 없이 gas 포함되면 gas
    
    2. nan인경우 많은 쪽으로 포함
    3. inhalation만 있는 경우 많은 쪽으로 포함
    4. aerosol, vapour, gas, dust, mist가 포함되어있지 않으면 많은 쪽으로 포함
'''

def check_nan(string):
    return string == string
        

def inhale_type(string):
    if check_nan(string):
        string = string.lower()
        
        if 'aerosol' in string:
            type_ = 'aerosol'
        
        elif 'dust' in string:
            type_ = 'aerosol'
        
        elif 'mist' in string:
            type_ = 'aerosol'
        
        elif 'aerosol' not in string and 'gas' in string:
            type_ = 'gas'
        
        elif 'aerosol' not in string and 'vapour' in string:
            type_ = 'vapour'
        
        elif 'aerosol' not in string and 'vapor' in string:
            type_ = 'vapour'
        
        elif 'aerosol' not in string and 'vaporization' in string:
            type_ = 'vapour'
        
        else:
            type_ = np.nan
    
    else:
        type_ = np.nan
        
    return type_


#%%
df = result.copy()

# generate inhale type
df['inhale type'] = df['Route of administration'].map(lambda x: inhale_type(x))

# if inhale type is nan then replace the value as maximal occurance of type
df['inhale type'][df['inhale type'].isna()] = df['inhale type'].value_counts(sort=True).index[0]

# save df
df.to_excel('tg413_raw.xlsx', header = True, index = False)
