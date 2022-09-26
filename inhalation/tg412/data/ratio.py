import openpyxl
import pandas as pd

vapour = pd.read_excel('vapour.xlsx')
aerosol = pd.read_excel('aerosol.xlsx')
gas = pd.read_excel('gas.xlsx')


print('count\n', vapour.category.value_counts().sort_index(),
      '\nratio\n', vapour.category.value_counts(normalize = True).sort_index().round(3))

print('count\n', aerosol.category.value_counts().sort_index(),
      '\nratio\n', aerosol.category.value_counts(normalize = True).sort_index().round(3))

print('count\n', gas.category.value_counts().sort_index(),
      '\nratio\n', gas.category.value_counts(normalize = True).sort_index().round(3))
