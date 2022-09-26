import sys
sys.path.append('../')

import openpyxl
import pandas as pd
from utils.smiles2fing import Smiles2Fing

vapour = pd.read_excel('vapour.xlsx')
aerosol = pd.read_excel('aerosol.xlsx')
gas = pd.read_excel('gas.xlsx')

# a, b = Smiles2Fing(gas.SMILES)

print('count\n', vapour.category.value_counts().sort_index(),
      '\nratio\n', vapour.category.value_counts(normalize = True).sort_index().round(3))

print('count\n', aerosol.category.value_counts().sort_index(),
      '\nratio\n', aerosol.category.value_counts(normalize = True).sort_index().round(3))

print('count\n', gas.category.value_counts().sort_index(),
      '\nratio\n', gas.category.value_counts(normalize = True).sort_index().round(3))
