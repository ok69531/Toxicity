import sys
sys.path.append('../')

import pandas as pd
from utils.smiles2fing import Smiles2Fing


vapour = pd.read_excel('vapour.xlsx')
aerosol = pd.read_excel('aerosol.xlsx')
gas = pd.read_excel('gas.xlsx')

v_drop_idx, fingerprints = Smiles2Fing(vapour.SMILES)
v_y = vapour.category.drop(v_drop_idx).reset_index(drop = True)

a_drop_idx, fingerprints = Smiles2Fing(aerosol.SMILES)
a_y = aerosol.category.drop(a_drop_idx).reset_index(drop = True)

g_drop_idx, fingerprints = Smiles2Fing(gas.SMILES)
g_y = gas.category.drop(g_drop_idx).reset_index(drop = True)


print('count\n', v_y.value_counts().sort_index(),
      '\nratio\n', v_y.value_counts(normalize = True).sort_index().round(3))

print('count\n', a_y.value_counts().sort_index(),
      '\nratio\n', a_y.value_counts(normalize = True).sort_index().round(3))

print('count\n', g_y.value_counts().sort_index(),
      '\nratio\n', g_y.value_counts(normalize = True).sort_index().round(3))
