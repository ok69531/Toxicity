import openpyxl
import pandas as pd
from .smiles2fing import Smiles2Fing


def load_data():
    df = pd.read_excel('../data/geno.xlsx')
    
    drop_idx, fingerprints = Smiles2Fing(df.SMILES)
    
    y = df.Genotoxicity.drop(drop_idx).replace({'negative': 0, 'positive': 1}).reset_index(drop = True)
    
    return fingerprints, y
