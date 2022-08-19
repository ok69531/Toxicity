import openpyxl
import pandas as pd
from smiles2fing import Smiles2Fing


def load_data(inhale_type):
    df = pd.read_excel('../data/' + inhale_type + '.xlsx')
    
    drop_idx, fingerprints = Smiles2Fing(df.SMILES)
    
    x = pd.concat([df.time.drop(drop_idx).reset_index(drop = True),
                   fingerprints],
                  axis = 1)
    y = df.category.drop(drop_idx)
    
    return x, y
