import openpyxl
import pandas as pd
from .smiles2fing import Smiles2Fing


def load_data():
    df = pd.read_excel('../data/noael.xlsx')
    
    drop_idx, fingerprints = Smiles2Fing(df.SMILES)
    
    y = df.category.drop(drop_idx)
    
    return fingerprints, y
