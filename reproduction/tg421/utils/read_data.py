import openpyxl
import pandas as pd
from .smiles2fing import Smiles2Fing


def load_data(admin_type):
    df = pd.read_excel('../data/' + admin_type + '.xlsx')
    
    drop_idx, fingerprints = Smiles2Fing(df.SMILES)
    
    y = df.value.drop(drop_idx).reset_index(drop = True)
    
    return fingerprints, y
