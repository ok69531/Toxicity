import openpyxl
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score
)

topkat = pd.read_excel('C:/Users/SOYOUNG/Desktop/tg403_topkat.xlsx', index_col='#')
topkat.index = topkat.index - 1
topkat = topkat.sort_index(axis = 0)

topkat_na_idx = topkat[topkat['예측결과'].isna() == True].index.tolist()

vapour_lgb = pd.read_excel('../predict/pred_result/vapour_lgb.xlsx')
aerosol_rf = pd.read_excel('../predict/pred_result/aerosol_rf.xlsx')
gas_rf = pd.read_excel('../predict/pred_result/gas_rf.xlsx')

ml_na_idx = vapour_lgb[vapour_lgb['pred'].isna() == True].index.tolist()
na_idx = topkat_na_idx + ml_na_idx

topkat = topkat.drop(na_idx).reset_index(drop = True)
vapour_lgb = vapour_lgb.drop(na_idx).reset_index(drop = True)
aerosol_rf = aerosol_rf.drop(na_idx).reset_index(drop = True)
gas_rf = gas_rf.drop(na_idx).reset_index(drop = True)

topkat.UNIT.unique()
topkat.UNIT.value_counts()

def unit_transform(value, unit):
    if unit == 'mg/m3/H':
        return value
    elif unit == 'g/m3/H':
        return value * 1000
    elif unit == 'ug/m3/H':
        return value * 0.001
    elif unit == 'ng/m3/H':
        return value * 1e-6
    elif unit == 'pg/m3/H':
        return value * 1e-9
    
topkat['pred'] = list(map(unit_transform, topkat.예측결과, topkat.UNIT))

topkat['vapour'] = pd.cut(topkat['pred'], bins = [0, 0.5, 2.0, 10.0, 20.0, np.infty], labels = range(5))
topkat['aerosol'] = pd.cut(topkat['pred'], bins = [0, 0.05, 0.5, 1.0, 5.0, np.infty], labels = range(5))
topkat['gas'] = pd.cut(topkat['pred'], bins = [0, 100, 500, 2500, 20000, np.infty], labels = range(5))

pd.crosstab(vapour_lgb.pred.astype(int), topkat.vapour, rownames=['ML'], colnames = ['TOPKAT'], margins = True)
pd.crosstab(aerosol_rf.pred.astype(int), topkat.aerosol, rownames=['ML'], colnames = ['TOPKAT'], margins = True)
pd.crosstab(gas_rf.pred.astype(int), topkat.vapour, rownames=['ML'], colnames = ['TOPKAT'], margins = True)

# precision_score(vapour_lgb.pred.astype(int), topkat.vapour, average = 'macro')
# recall_score(vapour_lgb.pred.astype(int), topkat.vapour, average = 'macro')
# accuracy_score(vapour_lgb.pred.astype(int), topkat.vapour)
# f1_score(vapour_lgb.pred.astype(int), topkat.vapour, average = 'macro')
