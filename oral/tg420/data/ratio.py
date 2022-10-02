import sys
sys.path.append('../')

import openpyxl
import pandas as pd
from utils.smiles2fing import Smiles2Fing

oral = pd.read_excel('oral.xlsx')

# a, b = Smiles2Fing(oral.SMILES)

print('count\n', oral.category.value_counts().sort_index(),
      '\nratio\n', oral.category.value_counts(normalize = True).sort_index().round(3))
