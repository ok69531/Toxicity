import sys
sys.path.append('../')

import openpyxl
import pandas as pd
from utils.smiles2fing import Smiles2Fing

geno = pd.read_excel('geno.xlsx')

# a, b = Smiles2Fing(geno.SMILES)

print('count\n', geno.Genotoxicity.value_counts().sort_index(),
      '\nratio\n', geno.Genotoxicity.value_counts(normalize = True).sort_index().round(3))
