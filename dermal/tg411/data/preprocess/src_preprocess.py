#%%
import re
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup


#%%
with open('tg411_page_src.json', 'r') as file:
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
# save df
result.to_excel('tg411_raw.xlsx', header = True, index = False)
