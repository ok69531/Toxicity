import pandas as pd

try: 
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit-pypi"])
    # subprocess.check_call([sys.executable, "-m", "conda", "install", "rdkit", "-c conda-forge"])
    
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys


def Smiles2Fing(smiles):
    ms_tmp = [Chem.MolFromSmiles(i) for i in smiles]
    ms_none_idx = [i for i in range(len(ms_tmp)) if ms_tmp[i] == None]
    
    ms = list(filter(None, ms_tmp))
    
    maccs = [MACCSkeys.GenMACCSKeys(i) for i in ms]
    maccs_bit = [i.ToBitString() for i in maccs]
    
    fingerprints = pd.DataFrame({'maccs': maccs_bit})
    fingerprints = fingerprints['maccs'].str.split(pat = '', n = 167, expand = True)
    fingerprints.drop(fingerprints.columns[0], axis = 1, inplace = True)
    
    colname = ['maccs_' + str(i) for i in range(1, 168)]
    fingerprints.columns = colname
    fingerprints = fingerprints.astype(int).reset_index(drop = True)
    
    return ms_none_idx, fingerprints
