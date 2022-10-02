import openpyxl
import pandas as pd
from .smiles2fing import Smiles2Fing


def load_data():
    df = pd.read_excel('../data/oral.xlsx')
    
    drop_idx, fingerprints = Smiles2Fing(df.SMILES)
    
    y = df.category.drop(drop_idx).reset_index(drop = True)
    
    return fingerprints, y
