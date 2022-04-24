'''
    Box-Cox ?
'''
#%%
import time
import random
import openpyxl

import pandas as pd
import numpy as np 

from tqdm import tqdm
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys, EState
from rdkit.Chem.EState import Fingerprinter
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import plot_precision_recall_curve, roc_curve, auc, confusion_matrix

from lightgbm import LGBMRegressor

pd.set_option('mode.chained_assignment', None)


#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print('=========================================')
print(device_lib.list_local_devices())
tf.debugging.set_log_device_placement(False)

np.random.seed(1)
tf.random.set_seed(1)


#%%
path = 'C:/Users/SOYOUNG/Desktop/toxic/data/'
chems_with_smiles_tmp = pd.read_excel(path + 'chems_with_smiles.xlsx', header = 0)
chems_with_smiles = chems_with_smiles_tmp.copy()

# didnt_work_idx = chems_with_smiles['SMILES'][chems_with_smiles['SMILES'] == 'Did not work'].index
# chems_with_smiles.drop(didnt_work_idx, axis = 0, inplace = True)
# chems_with_smiles.reset_index(drop = True, inplace = True)
# chems_with_smiles.shape


#%%
lc50_tmp = pd.read_excel(path + 'oecd_echemportal/Preprocessed data/inhalation403_lc50.xlsx', sheet_name = 'Sheet1')
lc50 = lc50_tmp.copy()

lc50['unit'].unique()
lc50['unit'].isna().sum()

lc50 = lc50[lc50['unit'].notna()]


casrn_drop_idx = lc50['CasRN'][lc50['CasRN'] == '-'].index
lc50.drop(casrn_drop_idx, inplace = True)


lc50['mg_per_L'] = np.nan

for i in tqdm(lc50.index):
    if ((lc50['unit'][i] == 'mg/L') or (lc50['unit'][i] == 'ppm')):
        lc50['mg_per_L'][i] = lc50['lower_value'][i]
    else:
        lc50['mg_per_L'][i] = lc50['lower_value'][i]*0.001


lc50_mean = lc50.groupby('CasRN')['mg_per_L'].mean().reset_index()
lc50_mean = lc50.groupby('CasRN')['mg_per_L'].min().reset_index()

# lc50_mean = lc50.groupby(['CasRN', 'Chemical_Name'])['mg_per_L'].mean().reset_index()
# lc50_mean['CasRN'].value_counts()
# lc50_mean[lc50_mean['CasRN'] == '1189173-49-6']

lc50_mean['mg_per_L'].describe()



#%%
data = pd.merge(lc50_mean, chems_with_smiles[['CasRN', 'SMILES']], on = 'CasRN', how = 'left').reset_index(drop  = True)
# data = pd.merge(lc50_mean, chems_with_smiles[['CasRN', 'SMILES']], on = 'CasRN', how = 'left').dropna().reset_index(drop  = True)

len(data['CasRN'].unique())
# data['SMILES'].isna().sum()

didnt_work_idx = data[data['SMILES'] == 'Did not work'].index
data = data.drop(didnt_work_idx).reset_index(drop = True)


#%%
# from urllib.request import urlopen
# from urllib.parse import quote 

# smiles_na_idx = data[data['SMILES'].isna()].index

# def CIRconvert(ids):
#     try:
#         url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
#         ans = urlopen(url).read().decode('utf8')
#         return ans
#     except:
#         return 'Did not work'


# data_smiles = data.iloc[smiles_na_idx]
# data_smiles['SMILES'] = [CIRconvert(data_smiles['CasRN'].iloc[i]) for i in tqdm(range(len(smiles_na_idx)))]


# chems_with_smiles = pd.concat([chems_with_smiles, data_smiles[['CasRN', 'SMILES']]], axis = 0)
# chems_with_smiles.to_excel(path + 'chems_with_smiles.xlsx', index = False, header = True)


#%%
''' fingerprint 변환 '''
# https://www.rdkit.org/docs/GettingStartedInPython.html
# https://www.programmersought.com/article/18903914496/


smiles = data['SMILES']
ms_tmp = [Chem.MolFromSmiles(i) for i in tqdm(smiles)]
ms_none_idx = [i for i in range(len(ms_tmp)) if ms_tmp[i] == None]
len(ms_none_idx)
ms = list(filter(None, ms_tmp))

macc = [MACCSkeys.GenMACCSKeys(i) for i in tqdm(ms)]
bit = [i.ToBitString() for i in tqdm(macc)]

fingerprint = pd.DataFrame(list(bit[0])).transpose().astype(float)

for i in tqdm(range(1, len(bit))):
    fingerprint_tmp = pd.DataFrame(list(bit[i])).transpose().astype(float)
    fingerprint = pd.concat([fingerprint, fingerprint_tmp], ignore_index = True)

fingerprint = fingerprint.astype(int)
fingerprint.dtypes.unique()


#%%
num_uniq = fingerprint.nunique()
drop_col_idx = num_uniq[num_uniq == 1].index
fingerprint.drop(drop_col_idx, axis = 1, inplace = True)


#%%
from mordred import Calculator, descriptors, ABCIndex, AcidBase, Weight, ZagrebIndex, WienerIndex

calc = Calculator(descriptors, ignore_3D = False)
# calc = Calculator([
#     ABCIndex,  # register all presets of descriptors in module (register ABC and ABCGG)
#     AcidBase.AcidicGroupCount,  # register all presets of the descriptor (register nAcid)
#     Weight.Weight(),  # register the descriptor (MW)
#     ZagrebIndex.ZagrebIndex(1, 1),  # Zagreb1
#     WienerIndex.WienerIndex(False),  # WPath
# ], ignore_3D = False)

smi_idx = list(set(range(len(smiles))) - set(ms_none_idx))
mol = [Chem.MolFromSmiles(smiles[i]) for i in smi_idx]

descriptor = pd.DataFrame()
for i in tqdm(range(len(mol))):
    des_tmp = pd.DataFrame().from_dict(calc(mol[i]).asdict(), orient = 'index').transpose().astype(float)
    descriptor = pd.concat([descriptor, des_tmp], axis = 0)


descriptor.reset_index(drop = True, inplace = True)

des_na_sum = descriptor.isna().sum()
des_col = des_na_sum[des_na_sum == 0].index
descriptor = descriptor[des_col]

descriptor.isna().sum().sum()
descriptor.dtypes.unique()


#%%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(descriptor)
scaler.data_min_
scaler.data_max_

scaled_descriptor = pd.DataFrame(scaler.transform(descriptor))
scaled_descriptor.columns = descriptor.columns


#%%
x = pd.concat([scaled_descriptor, fingerprint], axis = 1)
# y = data['mg_per_L'].drop(ms_none_idx, axis = 0).reset_index(drop = True)
y = np.log(data['mg_per_L'].drop(ms_none_idx, axis = 0).reset_index(drop = True))

# y = y[y <= 1]
# y = y[(y >= 1) & (y <= 5)]
# len(y)
# y = np.log(y[(y >= 1) & (y <= 10)])
plt.hist(y); plt.show()


#%%
from sklearn.linear_model import Lasso

lasso = Lasso(alpha = 0.01)
lasso.fit(x, y)
lasso_pred = lasso.predict(x)

print('mae: ', mean_absolute_error(y, lasso_pred))
print('mse: ', mean_squared_error(y, lasso_pred))


t = np.linspace(min(y), max(y), 100)
plt.figure(figsize = (5, 5))
plt.scatter(y, lasso_pred, alpha = 0.5)
plt.plot(t, t, color='darkorange', linewidth = 2)
plt.xlabel('true data', fontsize = 10)
plt.ylabel('prediction', fontsize = 10)
plt.show()

lasso_coef = lasso.coef_
x = x.iloc[:, np.where(lasso_coef != 0)[0]]


#%%
train_idx = random.sample(list(y.index), int(len(y) * 0.7))
test_idx = list(set(y.index) - set(train_idx))


x_train = x.iloc[train_idx]; x_test = x.iloc[test_idx]
# des_train = descriptor.iloc[train_idx]; des_test = descriptor.iloc[test_idx]
y_train = y.loc[train_idx]; y_test = y.loc[test_idx]


#%%
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
random_grid


rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
s1 = time.time()
rf_random.fit(x_train, y_train)
time.time() - s1
rf_random.best_params_


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(x_train, y_train)
base_accuracy = evaluate(base_model, x_test, y_test)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, x_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


#%%
rf = RandomForestRegressor(random_state = 1)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)

print('mae: ', mean_absolute_error(y_test, pred))
print('mse: ', mean_squared_error(y_test, pred))


t = np.linspace(min(y_test), max(y_test), 100)
plt.figure(figsize = (5, 5))
plt.scatter(y_test, pred, alpha = 0.5)
plt.plot(t, t, color='darkorange', linewidth = 2)
plt.xlabel('true data', fontsize = 10)
plt.ylabel('prediction', fontsize = 10)
plt.show()


# t = np.linspace(np.exp(min(y_test)), np.exp(max(y_test)), 100)
# plt.figure(figsize = (5, 5))
# plt.scatter(np.exp(y_test), np.exp(pred), alpha = 0.5)
# plt.plot(t, t, color='darkorange', linewidth = 2)
# plt.xlabel('true data', fontsize = 10)
# plt.ylabel('prediction', fontsize = 10)
# plt.show()


#%%
lgb = LGBMRegressor(random_state = 0, n_estimators = 1000)
lgb.fit(x_train, y_train)
lgb_pred = lgb.predict(x_test)

print('mae: ', mean_absolute_error(y_test, lgb_pred))
print('mse: ', mean_squared_error(y_test, lgb_pred))

t = np.linspace(min(y_test), max(y_test), 100)
plt.figure(figsize = (5, 5))
plt.scatter(y_test, pred, alpha = 0.5)
plt.plot(t, t, color = 'darkorange', linewidth = 2)
plt.xlabel('true data', fontsize = 10)
plt.ylabel('prediction', fontsize = 10)
plt.show()
plt.close()


# t = np.linspace(np.exp(min(y_test)), np.exp(max(y_test)), 100)
# plt.figure(figsize = (5, 5))
# plt.scatter(np.exp(y_test), np.exp(pred), alpha = 0.5)
# plt.plot(t, t, color = 'darkorange', linewidth = 2)
# plt.xlabel('true data', fontsize = 10)
# plt.ylabel('prediction', fontsize = 10)
# plt.show()
# plt.close()


#%%
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

y_train = tf.cast(y_train, tf.float32)
y_test = tf.cast(y_test, tf.float32)


#%%
input1 = layers.Input((x_train.shape[1]))

dense1 = layers.Dense(30, activation = 'relu')
# dense2 = layers.Dense(10)
dense3 = layers.Dense(1)

# yhat = dense3(dense2(dense1(input1)))
yhat = dense3(dense1(input1))
# yhat = dense3(input1)

model = K.models.Model(input1, yhat)
model.summary()


#%%
adam = K.optimizers.Adam(0.005)
# sgd = K.optimizers.SGD(0.001)
mae = K.losses.MeanAbsoluteError()

model.compile(optimizer = adam, loss = mae, metrics = ['mse'])
result = model.fit(x_train, y_train, epochs = 1000, batch_size = len(y_train))


#%%
# plt.plot(result.history['loss'], label='training')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()


#%%
pred = model.predict(x_test)
print(mae(y_test, pred).numpy())

t = np.linspace(min(y_test), max(y_test), 100)
plt.figure(figsize = (5, 5))
plt.scatter(y_test, pred, alpha = 0.5)
plt.plot(t, t, color='darkorange', linewidth = 2)
plt.xlabel('true data', fontsize = 10)
plt.ylabel('prediction', fontsize = 10)
plt.show()

