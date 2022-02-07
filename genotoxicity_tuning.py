#%%
import os
import time
import random
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import lightgbm
from lightgbm import LGBMClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")



#%%
path = 'C:/Users/SOYOUNG/Desktop/toxic/data/'

genotoxicity_tmp = pd.read_excel(path + 'genotoxicity/genotoxicity_input.xlsx')

# MACCS, ECFP(Extended fingerprints for smiles), PubChem, Estate, Morgan
fingerprints_tmp = genotoxicity_tmp.filter(regex = 'fingerprint|Fingerprint|(Bin)')
fingerprints_tmp.columns


#%%
# files = os.listdir(path + 'oecd_echemportal/Data tmp/')[1:-2]
# df = pd.DataFrame()

# for file in files:
#     if file.endswith('.xlsx'):
#         df = df.append(pd.read_excel(path + 'oecd_echemportal/Data tmp/' + file), ignore_index = True)

# df.drop(df.columns[3:], axis = 1).dropna().drop_duplicates(subset = ['Chemical_Name', 'CasRN', 'Genotoxicity'])

# len(set(genotoxicity_tmp['CAS RN']) - set(df['CasRN']))



#%%
''' 
    1. fingerprint만 사용해서 모델 성능 확인 using cverror -> 어떤 fingerprint의 성능이 가장 좋은지 확인
    2. 위에서 선택된 fingerprint + descriptor parameter tuning
'''

# y_tmp = genotoxicity_tmp['Genotoxicity(0.5기준)']

# y = pd.Series([0 if y_tmp[i] == 'N' else 1 for i in range(genotoxicity_tmp.shape[0])])
# y.value_counts()
# y.value_counts(normalize = True).plot(kind = 'bar'); plt.show()


# # MACCS fingerprints
# fingerprints_tmp['MACCS fingerprints for SMILES (Bin)'].isna().sum()
# maccs_len = len(list(fingerprints_tmp['MACCS fingerprints for SMILES (Bin)'][0]))
# maccs_na_idx = np.where(fingerprints_tmp['MACCS fingerprints for SMILES (Bin)'].isna() == True)

# maccs_y = y.drop(maccs_na_idx[0]).reset_index(drop = True)

# maccs_colname = ['maccs_' + str(i) for i in range(1, maccs_len + 1)]
# maccs = fingerprints_tmp['MACCS fingerprints for SMILES (Bin)'].drop(maccs_na_idx[0]).reset_index(drop = True)
# maccs = maccs.str.split(pat = '', n = maccs_len, expand = True)
# maccs.drop(maccs.columns[0], axis = 1, inplace = True)
# maccs = maccs.astype(int)
# maccs.columns = maccs_colname



# # ECFP(Extended fingerprints for smiles) fingerprints
# fingerprints_tmp['Extended fingerprints for SMILES (Bin)'].isna().sum()
# ecfp_len = len(list(fingerprints_tmp['Extended fingerprints for SMILES (Bin)'][0]))
# ecfp_na_idx = np.where(fingerprints_tmp['Extended fingerprints for SMILES (Bin)'].isna() == True)

# ecfp_y = y.drop(ecfp_na_idx[0]).reset_index(drop = True)

# ecfp_colname = ['ecfp_' + str(i) for i in range(1, ecfp_len + 1)]
# ecfp = fingerprints_tmp['Extended fingerprints for SMILES (Bin)'].drop(ecfp_na_idx[0]).reset_index(drop = True)
# ecfp = ecfp.str.split(pat = '', n = ecfp_len, expand = True)
# ecfp.drop(ecfp.columns[0], axis = 1, inplace = True)
# ecfp = ecfp.astype(int)
# ecfp.columns = ecfp_colname



# # PubChem
# fingerprints_tmp['Pubchem fingerprints for SMILES (Bin)'].isna().sum()
# pubchem_len = len(list(fingerprints_tmp['Pubchem fingerprints for SMILES (Bin)'][0]))
# # pubchem_na_idx = np.where(fingerprints_tmp['Pubchem fingerprints for SMILES (Bin)'].isna() == True)

# pubchem_y = y

# pubchem_colname = ['pubchem_' + str(i) for i in range(1, pubchem_len + 1)]
# pubchem = fingerprints_tmp['Pubchem fingerprints for SMILES (Bin)'].str.split(pat = '', n = pubchem_len, expand = True)
# pubchem.drop(pubchem.columns[0], axis = 1, inplace = True)
# pubchem = pubchem.astype(int)
# pubchem.columns = pubchem_colname



# # Estate fingerprints
# fingerprints_tmp['EState fingerprints for SMILES (Bin)'].isna().sum()
# estate_len = len(list(fingerprints_tmp['EState fingerprints for SMILES (Bin)'][0]))

# estate_y = y

# estate_colname = ['estate_' + str(i) for i in range(1, estate_len + 1)]
# estate = fingerprints_tmp['EState fingerprints for SMILES (Bin)'].str.split(pat = '', n = estate_len, expand = True)
# estate.drop(estate.columns[0], axis = 1, inplace = True)
# estate = estate.astype(int)
# estate.columns = estate_colname



# # Morgan fingerprints
# fingerprints_tmp['Morgan (Bin)'].isna().sum()
# morgan_len = len(list(fingerprints_tmp['Morgan (Bin)'][0]))
# morgan_na_idx = np.where(fingerprints_tmp['Morgan (Bin)'].isna() == True)

# morgan_y = y.drop(morgan_na_idx[0]).reset_index(drop = True)


# morgan_colname = ['morgan_' + str(i) for i in range(1, morgan_len + 1)]
# morgan = fingerprints_tmp['Morgan (Bin)'].drop(morgan_na_idx[0]).reset_index(drop = True)
# morgan = morgan.str.split(pat = '', n = morgan_len, expand = True)
# morgan.drop(morgan.columns[0], axis = 1, inplace = True)
# morgan = morgan.astype(int)
# morgan.columns = morgan_colname



# # Circular fingerprints for SMILES (Bin)
# fingerprints_tmp['Circular fingerprints for SMILES (Bin)'].isna().sum()
# circular_len = len(list(fingerprints_tmp['Circular fingerprints for SMILES (Bin)'][0]))

# circular_y = y

# circular_colname = ['circular_' + str(i) for i in range(1, circular_len + 1)]
# circular = fingerprints_tmp['Circular fingerprints for SMILES (Bin)'].str.split(pat = '', n = circular_len, expand = True)
# circular.drop(circular.columns[0], axis = 1, inplace = True)
# circular = circular.astype(int)
# circular.columns = circular_colname



# # Standard fingerprints for SMILES (Bin)
# fingerprints_tmp['Standard fingerprints for SMILES (Bin)'].isna().sum()
# stand_len = len(list(fingerprints_tmp['Standard fingerprints for SMILES (Bin)'][0]))
# stand_na_idx = np.where(fingerprints_tmp['Standard fingerprints for SMILES (Bin)'].isna() == True)

# stand_y = y.drop(stand_na_idx[0]).reset_index(drop = True)

# stand_colname = ['stand_' + str(i) for i in range(1, ecfp_len + 1)]
# stand = fingerprints_tmp['Standard fingerprints for SMILES (Bin)'].drop(stand_na_idx[0]).reset_index(drop = True)
# stand = stand.str.split(pat = '', n = stand_len, expand = True)
# stand.drop(stand.columns[0], axis = 1, inplace = True)
# stand = stand.astype(int)
# stand.columns = stand_colname


# # RDKit fingerprints
# fingerprints_tmp['RDKit (Bin)'].isna().sum()
# rdkit_len = len(list(fingerprints_tmp['RDKit (Bin)'][0]))
# rdkit_na_idx = np.where(fingerprints_tmp['RDKit (Bin)'].isna() == True)

# rdkit_y = y.drop(rdkit_na_idx[0]).reset_index(drop = True)

# rdkit_colname = ['rdkit_' + str(i) for i in range(1, rdkit_len + 1)]
# rdkit = fingerprints_tmp['RDKit (Bin)'].drop(rdkit_na_idx[0]).reset_index(drop = True)
# rdkit = rdkit.str.split(pat = '', n = rdkit_len, expand = True)
# rdkit.drop(rdkit.columns[0], axis = 1, inplace = True)
# rdkit = rdkit.astype(int)
# rdkit.columns = rdkit_colname




#%%
fingerprint_data = genotoxicity_tmp[['Genotoxicity(0.5기준)', 
                                     'MACCS fingerprints for SMILES (Bin)',
                                     'Extended fingerprints for SMILES (Bin)', 
                                     'Pubchem fingerprints for SMILES (Bin)',
                                     'EState fingerprints for SMILES (Bin)',
                                     'Morgan (Bin)','Circular fingerprints for SMILES (Bin)',
                                     'Standard fingerprints for SMILES (Bin)', 
                                     'RDKit (Bin)']]

fingerprint_data = fingerprint_data.dropna().reset_index(drop = True)


#%%
y_tmp = fingerprint_data['Genotoxicity(0.5기준)']

y = pd.Series([0 if y_tmp[i] == 'N' else 1 for i in range(fingerprint_data.shape[0])])
y.value_counts()
y.value_counts(normalize = True).plot(kind = 'bar'); plt.show()


# MACCS fingerprint
maccs_len = len(list(fingerprint_data['MACCS fingerprints for SMILES (Bin)'][0]))
maccs_colname = ['maccs_' + str(i) for i in range(1, maccs_len + 1)]
maccs = fingerprint_data['MACCS fingerprints for SMILES (Bin)'].str.split(pat = '', n = maccs_len, expand = True)
maccs.drop(maccs.columns[0], axis = 1, inplace = True)
maccs = maccs.astype(int)
maccs.columns = maccs_colname



# ECFP(Extended fingerprints for smiles) fingerprints
ecfp_len = len(list(fingerprint_data['Extended fingerprints for SMILES (Bin)'][0]))
ecfp_colname = ['ecfp_' + str(i) for i in range(1, ecfp_len + 1)]
ecfp = fingerprint_data['Extended fingerprints for SMILES (Bin)'].str.split(pat = '', n = ecfp_len, expand = True)
ecfp.drop(ecfp.columns[0], axis = 1, inplace = True)
ecfp = ecfp.astype(int)
ecfp.columns = ecfp_colname



# PubChem
pubchem_len = len(list(fingerprint_data['Pubchem fingerprints for SMILES (Bin)'][0]))
pubchem_colname = ['pubchem_' + str(i) for i in range(1, pubchem_len + 1)]
pubchem = fingerprint_data['Pubchem fingerprints for SMILES (Bin)'].str.split(pat = '', n = pubchem_len, expand = True)
pubchem.drop(pubchem.columns[0], axis = 1, inplace = True)
pubchem = pubchem.astype(int)
pubchem.columns = pubchem_colname



# Estate fingerprints
estate_len = len(list(fingerprints_tmp['EState fingerprints for SMILES (Bin)'][0]))
estate_colname = ['estate_' + str(i) for i in range(1, estate_len + 1)]
estate = fingerprint_data['EState fingerprints for SMILES (Bin)'].str.split(pat = '', n = estate_len, expand = True)
estate.drop(estate.columns[0], axis = 1, inplace = True)
estate = estate.astype(int)
estate.columns = estate_colname



# Morgan fingerprints
morgan_len = len(list(fingerprint_data['Morgan (Bin)'][0]))
morgan_colname = ['morgan_' + str(i) for i in range(1, morgan_len + 1)]
morgan = fingerprint_data['Morgan (Bin)'].str.split(pat = '', n = morgan_len, expand = True)
morgan.drop(morgan.columns[0], axis = 1, inplace = True)
morgan = morgan.astype(int)
morgan.columns = morgan_colname



# Circular fingerprints for SMILES (Bin)
circular_len = len(list(fingerprint_data['Circular fingerprints for SMILES (Bin)'][0]))
circular_colname = ['circular_' + str(i) for i in range(1, circular_len + 1)]
circular = fingerprint_data['Circular fingerprints for SMILES (Bin)'].str.split(pat = '', n = circular_len, expand = True)
circular.drop(circular.columns[0], axis = 1, inplace = True)
circular = circular.astype(int)
circular.columns = circular_colname



# Standard fingerprints for SMILES (Bin)
stand_len = len(list(fingerprint_data['Standard fingerprints for SMILES (Bin)'][0]))
stand_colname = ['stand_' + str(i) for i in range(1, ecfp_len + 1)]
stand = fingerprint_data['Standard fingerprints for SMILES (Bin)'].str.split(pat = '', n = stand_len, expand = True)
stand.drop(stand.columns[0], axis = 1, inplace = True)
stand = stand.astype(int)
stand.columns = stand_colname


# RDKit fingerprints
rdkit_len = len(list(fingerprint_data['RDKit (Bin)'][0]))
rdkit_colname = ['rdkit_' + str(i) for i in range(1, rdkit_len + 1)]
rdkit = fingerprint_data['RDKit (Bin)'].str.split(pat = '', n = rdkit_len, expand = True)
rdkit.drop(rdkit.columns[0], axis = 1, inplace = True)
rdkit = rdkit.astype(int)
rdkit.columns = rdkit_colname



#%%
logit_lasso = LogisticRegression(random_state = 0, penalty = 'l1', solver = 'saga')
# kfold = KFold(random_state = 0, n_splits = 10, shuffle = True)
kfold = StratifiedKFold(random_state = 1, n_splits = 10, shuffle = True)


maccs_score = np.mean(cross_val_score(logit_lasso, maccs, y, cv = kfold, scoring = 'f1'))
ecfp_score = np.mean(cross_val_score(logit_lasso, ecfp, y, cv = kfold, scoring = 'f1'))
pubchem_score = np.mean(cross_val_score(logit_lasso, pubchem, y, cv = kfold, scoring = 'f1'))
estate_score = np.mean(cross_val_score(logit_lasso, estate, y, cv = kfold, scoring = 'f1'))
morgan_score = np.mean(cross_val_score(logit_lasso, morgan, y, cv = kfold, scoring = 'f1'))
circular_score = np.mean(cross_val_score(logit_lasso, circular, y, cv = kfold, scoring = 'f1'))
standard_score = np.mean(cross_val_score(logit_lasso, stand, y, cv = kfold, scoring = 'f1'))
rdkit_score = np.mean(cross_val_score(logit_lasso, rdkit, y, cv = kfold, scoring = 'f1'))


print('\n cross-val scores using maccs fingerprints: ', maccs_score,
      '\n cross-val scores using ecfp fingerprints: ', ecfp_score,
      '\n cross-val scores using pubchem fingerprints: ', pubchem_score,
      '\n cross-val scores using estate fingerprints: ', estate_score,
      '\n cross-val scores using morgan fingerprints: ', morgan_score,
      '\n cross-val scores using circular fingerprints: ', circular_score,
      '\n cross-val scores using standard fingerprints: ', standard_score,
      '\n cross-val scores using rdkit fingerprints: ', rdkit_score)



#%%
# logit_lasso = LogisticRegression(penalty = 'l1', solver = 'saga')
# kfold = KFold(n_splits = 10, shuffle = True, random_state = 0)

# maccs_score = np.mean(cross_val_score(logit_lasso, maccs, maccs_y, cv = kfold))
# ecfp_score = np.mean(cross_val_score(logit_lasso, ecfp, ecfp_y, cv = kfold))
# pubchem_score = np.mean(cross_val_score(logit_lasso, pubchem, pubchem_y, cv = kfold))
# estate_score = np.mean(cross_val_score(logit_lasso, estate, estate_y, cv = kfold))
# morgan_score = np.mean(cross_val_score(logit_lasso, morgan, morgan_y, cv = kfold))
# circular_score = np.mean(cross_val_score(logit_lasso, circular, circular_y, cv = kfold))
# standard_score = np.mean(cross_val_score(logit_lasso, stand, stand_y, cv = kfold))
# rdkit_score = np.mean(cross_val_score(logit_lasso, rdkit, rdkit_y, cv = kfold))


# print('\n cross-val scores using maccs fingerprints: ', maccs_score,
#       '\n cross-val scores using ecfp fingerprints: ', ecfp_score,
#       '\n cross-val scores using pubchem fingerprints: ', pubchem_score,
#       '\n cross-val scores using estate fingerprints: ', estate_score,
#       '\n cross-val scores using morgan fingerprints: ', morgan_score,
#       '\n cross-val scores using circular fingerprints: ', circular_score,
#       '\n cross-val scores using standard fingerprints: ', standard_score,
#       '\n cross-val scores using rdkit fingerprints: ', rdkit_score)





#%%
'''
    사용할 데이터: Pubchem fingerprint
'''
genotoxicity_tmp['Pubchem fingerprints for SMILES (Bin)'].isna().sum()

y_tmp = genotoxicity_tmp['Genotoxicity(0.5기준)']

y = pd.Series([0 if y_tmp[i] == 'N' else 1 for i in range(genotoxicity_tmp.shape[0])])
y.value_counts()
# y.value_counts(normalize = True).plot(kind = 'bar'); plt.show()


fingerprint = genotoxicity_tmp['Pubchem fingerprints for SMILES (Bin)'].str.split(pat = '', n = pubchem_len, expand = True)
fingerprint.drop(fingerprint.columns[0], axis = 1, inplace = True)
fingerprint = fingerprint.astype(int)
fingerprint.columns = pubchem_colname



#%%
# from rdkit import Chem
# from mordred import Calculator, descriptors

# calc = Calculator(descriptors, ignore_3D = False)
# descriptor = [calc(Chem.MolFromSmiles(x)).asdict() for x in tqdm(genotoxicity_tmp['SMILES'])]


#%%
descriptors_tmp = genotoxicity_tmp.drop(fingerprints_tmp.columns, axis = 1).iloc[:, 5:]

descriptors_tmp.isna().sum()
des_na = descriptors_tmp.isna().sum()[descriptors_tmp.isna().sum() != 0]
descriptor = descriptors_tmp.drop(des_na.index, axis = 1)

descriptor.dtypes.unique()
des_object_idx = descriptor.dtypes[descriptor.dtypes == 'O'].index
descriptor[des_object_idx]

descriptor.drop(des_object_idx, axis = 1, inplace = True)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(descriptor)
scaler.data_min_
scaler.data_max_

scaled_descriptor = pd.DataFrame(scaler.transform(descriptor))
scaled_descriptor.columns = descriptor.columns


data = pd.concat([fingerprint, scaled_descriptor], axis = 1)


#%%
random.seed(0)
train_idx = random.sample(range(data.shape[0]), round(data.shape[0] * 0.8))
test_idx = list(set(range(data.shape[0])) - set(train_idx))


fing_train = fingerprint.iloc[train_idx]; fing_test = fingerprint.iloc[test_idx]
data_train = data.iloc[train_idx]; data_test = data.iloc[test_idx]
y_train = y.iloc[train_idx]; y_test = y.iloc[test_idx]



from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = 0)
fing_over_train, y_over_train = smote.fit_resample(fing_train, y_train)
data_over_train, y_over_train = smote.fit_resample(data_train, y_train)



#%%
'''
    Model Fitting
'''
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

logistic = LogisticRegression(random_state = 0, penalty = 'l1')

# C = 1, max_iter = 100
logit_params = {'C': [10, 15, 20, 25],
                'solver': ['liblinear', 'saga'],
                'max_iter': [25, 30, 40, 50, 60]}

score = ['precision', 'recall', 'f1', 'roc_auc', 'accuracy']


start = time.time()
logit_clf = GridSearchCV(logistic, logit_params, scoring = score, refit = 'f1')
# logit_clf.fit(X = fing_train, y = y_train)
# logit_clf.fit(X = fing_over_train, y = y_over_train)
# logit_clf.fit(X = data_train, y = y_train)
logit_clf.fit(X = data_over_train, y = y_over_train)
print('time: ', time.time() - start)

print(logit_clf.best_params_)

# logit_pred = logit_clf.predict(fing_test)
logit_pred = logit_clf.predict(data_test)


print('\nPrecision: ', precision_score(y_test, logit_pred),
      '\nRecall: ', recall_score(y_test, logit_pred),
      '\nF1 score: ', f1_score(y_test, logit_pred),
      '\nAUC: ', roc_auc_score(y_test, logit_pred),
      '\nAccuracy: ', accuracy_score(y_test, logit_pred),
      '\n', classification_report(y_test, logit_pred))


# plot_confusion_matrix(logit_clf, fing_test, y_test, normalize = 'true', cmap = plt.cm.Blues)
plot_confusion_matrix(logit_clf, data_test, y_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
plt.close()

'''
    fingerprint raw & raw: C = 5, max_iter = 30, solver = 'liblinear'
    fingerprint over & raw: C = 3, max_iter = 15, solver = 'liblinear'
    
    fingerprint+descriptor raw & raw: C = 7, max_iter = 10, solver = 'liblinear'
    fingerprint+descriptor over & raw: C = 10, max_iter = 25, solver = 'liblinear'
'''


#%%
rf = RandomForestClassifier(random_state = 0)

# n_estimators = 70, max_depth = None, min_samples_split = 2, min_samples_leaf = 1
rf_params = {'n_estimators': [70, 100, 130],
             'criterion': ['gini', 'entropy'],
             'max_depth': ['None', 2, 5, 8],
             'min_samples_split': [1, 2, 4, 6],
             'min_samples_leaf': [1, 2, 3]}


start = time.time()
rf_clf = GridSearchCV(rf, rf_params, scoring = score, refit = 'f1')

rf_clf.fit(X = fing_train, y = y_train)
# rf_clf.fit(X = fing_over_train, y = y_over_train)
# rf_clf.fit(X = data_train, y = y_train)
# rf_clf.fit(X = data_over_train, y = y_over_train)
print('time: ', time.time() - start)

print(logit_clf.best_params_)

rf_pred = rf_clf.predict(fing_test)
# rf_pred = rf_clf.predict(data_test)


print('\nPrecision: ', precision_score(y_test, rf_pred),
      '\nRecall: ', recall_score(y_test, rf_pred),
      '\nF1 score: ', f1_score(y_test, rf_pred),
      '\nAUC: ', roc_auc_score(y_test, rf_pred),
      '\nAccuracy: ', accuracy_score(y_test, rf_pred),
      '\n', classification_report(y_test, rf_pred))

plot_confusion_matrix(rf_clf, fing_test, y_test, normalize = 'true', cmap = plt.cm.Blues)
# plot_confusion_matrix(rf_clf, data_test, y_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
plt.close()


'''
    fingerprint raw & raw: n_estimators = 80, max_depth = 40, min_samples_split = 2, criterion = 
    fingerprint over & raw: n_estimators = 70, max_depth = 30, min_samples_split = 2, criterion = 'entropy'
    
    fingerprint+descriptor raw & raw: n_estimators = 120, max_depth = 30, min_samples_split = 2, criterion = 'entropy'
    fingerprint+descriptor over & raw: n_estimators = 120, max_depth = 30, min_samples_split = 2
'''




#%%
lgb = LGBMClassifier(random_state = 0, is_unbalance = True)


# num_leaves = 31, min_child_samples = 20, max_depth = -1, learning_rate = 0.1
lgb_params = {'num_leaves': [35, 40, 45, 50],
              'min_child_samples': [15, 20, 25],
              'learning_rate': [0.1, 0.05, 0.03, 0.01]}
# 0.1, 3, 110

start = time.time()
lgb_clf = GridSearchCV(lgb, lgb_params, scoring = score, refit = 'f1')
# lgb_clf.fit(X = fing_train, y = y_train)
# lgb_clf.fit(X = fing_over_train, y = y_over_train)
# lgb_clf.fit(X = data_train, y = y_train)
lgb_clf.fit(X = data_over_train, y = y_over_train)
print('time: ', time.time() - start)

print(lgb_clf.best_params_)

# lgb_pred = lgb_clf.predict(fing_test)
lgb_pred = lgb_clf.predict(data_test)

print('\nPrecision: ', precision_score(y_test, lgb_pred),
      '\nRecall: ', recall_score(y_test, lgb_pred),
      '\nF1 score: ', f1_score(y_test, lgb_pred),
      '\nAUC: ', roc_auc_score(y_test, lgb_pred),
      '\nAccuracy: ', accuracy_score(y_test, lgb_pred),
      '\n', classification_report(y_test, lgb_pred))

# plot_confusion_matrix(lgb_clf, fing_test, y_test, normalize = 'true', cmap = plt.cm.Blues)
plot_confusion_matrix(lgb_clf, data_test, y_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
plt.close()


'''
    fingerprint raw & raw: num_leaves = 50, min_child_samples = 20, learning_rate = 0.05
    fingerprint over & raw: num_leaves = 280, min_child_samples = 1, learning_rate = 0.1
    
    fingerprint+descriptor raw & raw: num_leaves = 40, min_child_samples = 20, learning_rate = 0.05
    fingerprint+descriptor over & raw: num_leaves = 50, min_child_samples = 25, learning_rate = 0.1
'''



#%%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree')

# n_neighbors = 5, leaf_size = 30, p = 2
knn_params = {'n_neighbors': [2, 5, 8],
             'leaf_size': [10, 20, 30, 40, 50],
             'p': [1, 2, 3]}


start = time.time()
knn_clf = GridSearchCV(knn, knn_params, scoring = score, refit = 'f1')

knn_clf.fit(X = fing_train, y = y_train)
# knn_clf.fit(X = fing_over_train, y = y_over_train)
# knn_clf.fit(X = data_train, y = y_train)
# knn_clf.fit(X = data_over_train, y = y_over_train)
print('time: ', time.time() - start)

print(knn_clf.best_params_)

knn_pred = knn_clf.predict(fing_test)
# knn_pred = knn_clf.predict(data_test)


print('\nPrecision: ', precision_score(y_test, knn_pred))
print('\nRecall: ', recall_score(y_test, knn_pred))
print('\nF1 score: ', f1_score(y_test, knn_pred))
print('\nAUC: ', roc_auc_score(y_test, knn_pred))
print('\nAccuracy: ', accuracy_score(y_test, knn_pred))
print('\n', classification_report(y_test, knn_pred))

plot_confusion_matrix(knn_clf, fing_test, y_test, normalize = 'true', cmap = plt.cm.Blues)
# plot_confusion_matrix(knn_clf, data_test, y_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
plt.close()


'''
    fingerprint raw & raw: n_neighbors = 5, leaf_size = 20, p = 1
    fingerprint over & raw: n_neighbors = 2, leaf_size = 15, p = 1
    
    fingerprint+descriptor raw & raw: n_neighbors = 5, leaf_size = 15, p = 1
    fingerprint+descriptor over & raw: n_neighbors = 2, leaf_size = 10, p = 1
'''



#%%
mlp = MLPClassifier(random_state = 0)

# activation = 'relu', solver = 'adam', learning_rate_init = 0.001
mlp_params = {'activation': ['tanh', 'relu'],
              'solver': ['sgd', 'adam'],
              'learning_rate_init': [0.001, 0.01]}

print('fingerpring and descriptor oversampling & raw')
start = time.time()
mlp_clf = GridSearchCV(mlp, mlp_params, scoring = score, refit = 'f1')
# mlp_clf.fit(X = fing_train, y = y_train)
# mlp_clf.fit(X = fing_over_train, y = y_over_train)
# mlp_clf.fit(X = data_train, y = y_train)
mlp_clf.fit(X = data_over_train, y = y_over_train)
print('time: ', time.time() - start)

print(mlp_clf.best_params_)

# mlp_pred = mlp_clf.predict(fing_test)
mlp_pred = mlp_clf.predict(data_test)


print('\nPrecision: ', precision_score(y_test, mlp_pred), 
      '\nRecall: ', recall_score(y_test, mlp_pred),
      '\nF1 score: ', f1_score(y_test, mlp_pred),
      '\nAUC: ', roc_auc_score(y_test, mlp_pred), 
      '\nAccuracy: ', accuracy_score(y_test, mlp_pred),
      '\n', classification_report(y_test, mlp_pred))

# plot_confusion_matrix(mlp_clf, fing_test, y_test, normalize = 'true', cmap = plt.cm.Blues)
plot_confusion_matrix(mlp_clf, data_test, y_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
plt.close()


'''
    fingerprint raw & raw: activation = 'tanh', solver = 'adam', learning_rate_init = 0.001
    fingerprint over & raw: activation = , solver = , learning_rate_init = 
    
    fingerprint+descriptor raw & raw: activation = 'relu', solver = 'sgd', learning_rate_init = 0.01
    fingerprint+descriptor over & raw: activation = 'tanh', solver = 'adam', learning_rate_init = 0.001
'''