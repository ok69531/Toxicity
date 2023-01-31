import sys
sys.path.append('../')

import openpyxl
import pandas as pd
from utils.smiles2fing import Smiles2Fing

df = pd.read_excel('oral.xlsx')
drop_idx, fingerprints = Smiles2Fing(df.SMILES)
y = df.category.drop(drop_idx).reset_index(drop = True)
    
print('count\n', y.value_counts().sort_index(),
      '\nratio\n', y.value_counts(normalize = True).sort_index().round(3))
