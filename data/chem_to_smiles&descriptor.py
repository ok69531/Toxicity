#%%
# !pip install xlrd
# !pip install openpyxl
# !pip install scikit-learn

import openpyxl
import pymysql
import os 

import pandas as pd
import numpy as np 

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from mordred import Calculator, descriptors
# from sklearn.feature_extraction import DictVectorizer

#%%
mydb = pymysql.connect(host = '127.0.0.1', user = 'root', password = '',
                       charset = 'utf8mb4', cursorclass = pymysql.cursors.DictCursor, 
                       db = 'pyctd')
conn = mydb.cursor()


conn.execute('SHOW tables;')
all_tabs = conn.fetchall()
[all_tabs[i]['Tables_in_pyctd'] for i in range(len(all_tabs))]


conn.execute('SELECT ChemicalName, ChemicalID, CasRN from allchems;')
chems = pd.DataFrame(conn.fetchall())

conn.execute('SELECT * FROM cd')
cd = pd.DataFrame(conn.fetchall())
# cd['toxicity'] = 1

conn.execute('SELECT * FROM alldiseases')
dis = pd.DataFrame(conn.fetchall())
pd.isna(dis['SlimMappings']).sum()
dis['SlimMappings'] = dis['SlimMappings'].fillna('unknown')

#%%
# os.chdir()
print('Current Working Directory', os.getcwd())

ctod_tmp = pd.ExcelFile('DtoC_score_10diseases.xlsx')
dis_id = ctod_tmp.sheet_names[4:]

for i in tqdm(range(len(dis_id))):
    if i == 0:
        ctod = pd.read_excel(ctod_tmp, dis_id[i], header = 3).iloc[:, :5]
        ctod['Disease Name'] = dis_id[i]
    else:
        tmp = pd.read_excel(ctod_tmp, dis_id[i], header = 3).iloc[:, :5]
        tmp['Disease Name'] = dis_id[i]
        ctod = pd.concat([ctod, tmp])

ctod.reset_index(inplace = True)
ctod.shape
len(ctod['Disease Name'].unique())
len(ctod['Chemical Name'].unique())


#%%
# Disease data에서 parent id(-> SlimMappings 이용해야 할듯?) 이용해서 호흡계, 신경계질병 확인하기

dis['category'] = [dis['SlimMappings'][i].split('|')[0] for i in range(dis.shape[0])]
dis[['DiseaseName', 'DiseaseID', 'SlimMappings', 'category']]

cd_join = pd.merge(cd, dis[['DiseaseID','category']], left_on='DiseaseID', right_on='DiseaseID', how='left')

cd_join[cd_join['ChemicalID'] == ctod['Chemical ID'][69]][['ChemicalName', 'ChemicalID', 'DiseaseName','DiseaseID', 'category']]


#%%
# CasRN to SMILES
from urllib.request import urlopen
from urllib.parse import quote 

def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return 'Did not work'


# chems_without_nan = chems.copy()
# chems_without_nan.dropna(subset = ["CasRN"], inplace=True)
# chems_without_nan.reset_index(inplace = True)
# chems_without_nan['SMILES'] = [CIRconvert(chems_without_nan['CasRN'][i]) for i in range(chems_without_nan.shape[0])]

# chems_without_nan.to_csv('chems_with_smiles.csv', sep = ',', header = True, index = False)

chems_without_nan = pd.read_csv('chems_with_smiles.csv')

didnt_work_idx = chems_without_nan['SMILES'][chems_without_nan['SMILES'] == 'Did not work'].index
# didnt_work = chems_without_nan.iloc[didnt_work_idx, :]
# CIRconvert(didnt_work['CasRN'][didnt_work_idx[1]])


chems_with_smiles = chems_without_nan.drop(didnt_work_idx, axis = 0)
chems_with_smiles.reset_index(drop = True, inplace = True)


# ------------------------------------------------------------------------------------------- #

#%%
# descriptor : https://mordred-descriptor.github.io/documentation/master/descriptors.html
# constitutional descriptor : https://www.epa.gov/sites/production/files/2015-05/documents/moleculardescriptorsguide-v102.pdf
#                             ---> 45개 

calc = Calculator(descriptors, ignore_3D = True)
# calc.descriptors

chem_to_dcp = chems_with_smiles[['ChemicalID', 'SMILES']]


for i in tqdm(range(100)):
    if i==0:
        result = calc(Chem.MolFromSmiles(chem_to_dcp['SMILES'][i])).asdict()
        # result_dict = result.drop_missing().asdict()
        
        dcp = pd.DataFrame(list(result.items())).transpose()
        dcp.columns = dcp.iloc[0, :]
        dcp = dcp.drop(0, axis = 0)
    else:
        result = calc(Chem.MolFromSmiles(chem_to_dcp['SMILES'][i])).asdict()
        
        dcp_tmp = pd.DataFrame(list(result.items())).transpose()
        dcp_tmp.columns = dcp_tmp.iloc[0, :]
        dcp_tmp = dcp_tmp.drop(0, axis = 0)
        
        dcp = dcp.append(dcp_tmp)

dcp.reset_index(drop = True, inplace = True)
dcp['ChemicalID'] = chem_to_dcp['ChemicalID'][:100]
dcp['SMILES'] = chem_to_dcp['SMILES'][:100]

dcp[list(dcp.columns[-2:]) + list(dcp.columns[:1613])]


# mol1 = Chem.MolFromSmiles('CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C')
# mol2 = Chem.MolFromSmiles('CCCC1=NC(=C2N1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)CC)OCC)C')
# Draw.MolsToGridImage([mol1, mol2])

# fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
# fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
# dist = 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)

#%%
