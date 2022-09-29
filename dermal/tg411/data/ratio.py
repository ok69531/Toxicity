import sys
sys.path.append('../')

import openpyxl
import pandas as pd
from utils.smiles2fing import Smiles2Fing

noael = pd.read_excel('noael.xlsx')

# a, b = Smiles2Fing(noael.SMILES)

print('count\n', noael.category.value_counts().sort_index(),
      '\nratio\n', noael.category.value_counts(normalize = True).sort_index().round(3))
