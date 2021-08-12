#%%
import pandas as pd
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys, RDKFingerprint


#%%
chems_with_smiles_tmp = pd.read_excel('C:/Users/SOYOUNG/Desktop/toxic/data/chems_with_smiles.xlsx', header = 0)
chems_with_smiles = chems_with_smiles_tmp.copy()

smiles = chems_with_smiles['SMILES']

#%%
ms_tmp = [Chem.MolFromSmiles(i) for i in tqdm(smiles)]
ms_na_idx = [i for i in range(len(ms_tmp)) if ms_tmp[i] == None]
len(ms_na_idx)
ms = list(filter(None, ms_tmp))

macc = [MACCSkeys.GenMACCSKeys(i) for i in tqdm(ms)]
bit = [i.ToBitString() for i in tqdm(macc)]

maccs = pd.DataFrame(list(bit[0])).transpose()

for i in tqdm(range(1, len(bit))):
    maccs_tmp = pd.DataFrame(list(bit[i])).transpose()
    maccs = pd.concat([maccs, maccs_tmp], ignore_index = True)

macc_smiles = pd.DataFrame({'CasRN' : chems_with_smiles['CasRN'].drop(ms_na_idx, axis = 0).reset_index(drop = True),
                            'SMILES' : smiles.drop(ms_na_idx, axis = 0).reset_index(drop = True)})
maccs_fingerprint = pd.concat([macc_smiles, maccs], axis = 1)
maccs_fingerprint.isna().sum().sum()

maccs_fingerprint.to_excel('maccs.xlsx', header = False, index = False)

#%%
# fp_tmp = [Chem.MolFromSmiles(x) for x in tqdm(smiles)]
# fp_na_idx = [i for i in range(len(fp_tmp)) if fp_tmp[i] == None]
# len(fp_na_idx)
# fp = list(filter(None, fp_tmp))

fp = [Chem.RDKFingerprint(i) for i in tqdm(ms)]
fp_bit = [i.ToBitString() for i in tqdm(fp)]

fps = pd.DataFrame(list(fp_bit[0])).transpose()

for i in tqdm(range(1, len(fp_bit))):
    fps_tmp = pd.DataFrame(list(fp_bit[i])).transpose()
    fps = pd.concat([fps, fps_tmp], ignore_index = True)


fps_smiles = pd.DataFrame({'CasRN' : chems_with_smiles['CasRN'].drop(ms_na_idx, axis = 0).reset_index(drop = True),
                           'SMILES' : smiles.drop(ms_na_idx, axis = 0).reset_index(drop = True)})
fps_fingerprint = pd.concat([fps_smiles, maccs], axis = 1)
fps_fingerprint.isna().sum().sum()

fps_fingerprint.to_excel('fps.xlsx', header = False, index = False)
