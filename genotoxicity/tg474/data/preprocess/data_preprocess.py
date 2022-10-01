#%%
import re
import cirpy
import openpyxl

import pandas as pd
import numpy as np 

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)


#%%
def check_endpoint(string):
    try:
        endpoint = re.findall('positive|negative', string)
        return endpoint[0]
    except:
        return np.nan


# 하나의 화합물이 여러 개의 결과를 갖을 때, 보수적으로 진행하기 위해 하나라도 positive 결과를 갖으면 Genotoxicity를 postivie로 지정
def extract_endpoint(casrn):
    length = len(geno_tmp.Genotoxicity[geno_tmp.CasRN == casrn].unique())
    
    if length == 1:
        return geno_tmp.Genotoxicity[geno_tmp.CasRN == casrn].unique()[0]
    
    elif length > 1:
        # count = geno_tmp.Genotoxicity[geno_tmp.CasRN == casrn].value_counts()
        return 'positive'


#%%
df_tmp = pd.read_excel('tg474_raw.xlsx', header = 0)

geno_tmp = df_tmp[['Chemical', 'CasRN', 'Genotoxicity']]
geno_tmp['Genotoxicity'] = geno_tmp.Genotoxicity.map(lambda x: check_endpoint(str(x)))
geno_tmp = geno_tmp[geno_tmp.Genotoxicity.notna()].reset_index(drop = True)

geno = geno_tmp[['Chemical', 'CasRN']].drop_duplicates(['CasRN']).reset_index(drop = True)
geno['Genotoxicity'] = geno.CasRN.map(lambda x: extract_endpoint(x))

# generate SMILES
tqdm.pandas()
geno['SMILES'] = geno.CasRN.progress_apply(lambda x: cirpy.resolve(x, 'smiles'))
geno.SMILES.isna().sum()
geno = geno[geno.SMILES.notna()].reset_index(drop = True)

geno.to_excel('../geno.xlsx', index = False, header = True)
