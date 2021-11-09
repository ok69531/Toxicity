#%%
import time
import random
import openpyxl
import warnings

import pandas as pd
import numpy as np 

from tqdm import tqdm
# from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys, EState
from rdkit.Chem.EState import Fingerprinter
from matplotlib import pyplot as plt

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import plot_precision_recall_curve, roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, cohen_kappa_score, plot_confusion_matrix


pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings("ignore")

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

lc50_tmp = pd.read_excel(path + 'oecd_echemportal/Preprocessed data/inhalation403_lc50.xlsx', sheet_name = 'Sheet1')
lc50 = lc50_tmp.copy()

data = pd.merge(lc50, chems_with_smiles[['CasRN', 'SMILES']], on = 'CasRN', how = 'left').reset_index(drop = True)
# data = pd.merge(lc50_mean, chems_with_smiles[['CasRN', 'SMILES']], on = 'CasRN', how = 'left').dropna().reset_index(drop  = True)

len(data['CasRN'].unique())
# data['SMILES'].isna().sum()

data['unit'].unique()
data['unit'].isna().sum()
data = data[lc50['unit'].notna()]

casrn_na_idx = data[data['CasRN'] == '-'].index
didnt_work_idx = data[data['SMILES'] == 'Did not work'].index
data = data.drop(list(casrn_na_idx) + list(didnt_work_idx)).reset_index(drop = True)


#%%
# ppm data
lc50_ppm_tmp = data[data['unit'] == 'ppm']
lc50_ppm = lc50_ppm_tmp.groupby(['CasRN', 'SMILES'])['lower_value'].mean().reset_index()
lc50_ppm.columns = list(lc50_ppm.columns[0:2]) + ['value']
lc50_ppm['value'].describe()

# mg/L data
lc50_mgl_tmp = data[data['unit'] != 'ppm']
lc50_mgl_tmp['value'] = [lc50_mgl_tmp['lower_value'][i] if lc50_mgl_tmp['unit'][i] is 'mg/L' else 
                         lc50_mgl_tmp['lower_value'][i]*0.001 for i in lc50_mgl_tmp.index]
lc50_mgl = lc50_mgl_tmp.groupby(['CasRN', 'SMILES'])['value'].mean().reset_index()
lc50_mgl['value'].describe()


#%%
# ppm MACCS fingerprint
ppm_smiles = lc50_ppm['SMILES']
ppm_ms_tmp = [Chem.MolFromSmiles(i) for i in tqdm(ppm_smiles)]
ms_none_idx = [i for i in range(len(ppm_ms_tmp)) if ppm_ms_tmp[i] == None]
len(ms_none_idx)
ppm_ms = list(filter(None, ppm_ms_tmp))

ppm_maccs = [MACCSkeys.GenMACCSKeys(i) for i in tqdm(ppm_ms)]
ppm_bit = [i.ToBitString() for i in tqdm(ppm_maccs)]

ppm_fingerprint = pd.DataFrame(list(ppm_bit[0])).transpose().astype(int)

for i in tqdm(range(1, len(ppm_bit))):
    fingerprint_tmp = pd.DataFrame(list(ppm_bit[i])).transpose().astype(int)
    ppm_fingerprint = pd.concat([ppm_fingerprint, fingerprint_tmp], ignore_index = True)

ppm_fingerprint.dtypes.unique()


ppm_fing_uniq = ppm_fingerprint.nunique()
ppm_fing_col_idx = ppm_fing_uniq[ppm_fing_uniq == 1].index
ppm_fingerprint.drop(ppm_fing_col_idx, axis = 1, inplace = True)


# mg/L MACCS fingerprint
mgl_smiles = lc50_mgl['SMILES']
mgl_ms_tmp = [Chem.MolFromSmiles(i) for i in tqdm(mgl_smiles)]
ms_none_idx = [i for i in range(len(mgl_ms_tmp)) if mgl_ms_tmp[i] == None]
len(ms_none_idx)
mgl_ms = list(filter(None, mgl_ms_tmp))

mgl_maccs = [MACCSkeys.GenMACCSKeys(i) for i in tqdm(mgl_ms)]
mgl_bit = [i.ToBitString() for i in tqdm(mgl_maccs)]

mgl_fingerprint = pd.DataFrame(list(mgl_bit[0])).transpose().astype(int)

for i in tqdm(range(1, len(mgl_bit))):
    fingerprint_tmp = pd.DataFrame(list(mgl_bit[i])).transpose().astype(int)
    mgl_fingerprint = pd.concat([mgl_fingerprint, fingerprint_tmp], ignore_index = True)

mgl_fingerprint.dtypes.unique()


mgl_fing_uniq = mgl_fingerprint.nunique()
mgl_fing_col_idx = mgl_fing_uniq[mgl_fing_uniq == 1].index
mgl_fingerprint.drop(mgl_fing_col_idx, axis = 1, inplace = True)


#%%
from mordred import Calculator, descriptors, ABCIndex, AcidBase, Weight, ZagrebIndex, WienerIndex
from sklearn.preprocessing import MinMaxScaler

# ppm descriptor
calc = Calculator(descriptors, ignore_3D = True)

ppm_descriptor = pd.DataFrame()
for i in tqdm(range(len(ppm_ms_tmp))):
    des_tmp = pd.DataFrame().from_dict(calc(ppm_ms_tmp[i]).asdict(), orient = 'index').transpose().astype(float)
    ppm_descriptor = pd.concat([ppm_descriptor, des_tmp], axis = 0)

ppm_descriptor.reset_index(drop = True, inplace = True)

ppm_des_na = ppm_descriptor.isna().sum()
ppm_des_col = ppm_des_na[ppm_des_na == 0].index
ppm_descriptor = ppm_descriptor[ppm_des_col]

ppm_des_uniq = ppm_descriptor.nunique()
ppm_des_drop_idx = ppm_des_uniq[ppm_des_uniq == 1].index
ppm_descriptor.drop(ppm_des_drop_idx, axis = 1, inplace = True)


ppm_scaler = MinMaxScaler()
ppm_scaler.fit(ppm_descriptor)
ppm_scaler.data_min_
ppm_scaler.data_max_


ppm_scaled_descriptor = pd.DataFrame(ppm_scaler.transform(ppm_descriptor))
ppm_scaled_descriptor.columns = ppm_descriptor.columns

ppm_x = pd.concat([ppm_fingerprint, round(ppm_scaled_descriptor, 5)], axis = 1)
ppm_y = pd.DataFrame({'value': lc50_ppm['value'],
                     'category': pd.cut(lc50_ppm['value'], bins = [0, 100, 500, 2500, 20000, np.infty], labels = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5'])})
ppm_y['category'].value_counts()
# LabelBinarizer().fit_transform(ppm_y['category'])


#%%
# mgl descriptor
calc = Calculator(descriptors, ignore_3D = True)

mgl_descriptor = pd.DataFrame()
for i in tqdm(range(len(mgl_ms))):
    des_tmp = pd.DataFrame().from_dict(calc(mgl_ms[i]).asdict(), orient = 'index').transpose().astype(float)
    mgl_descriptor = pd.concat([mgl_descriptor, des_tmp], axis = 0)

mgl_descriptor.reset_index(drop = True, inplace = True)

mgl_des_na = mgl_descriptor.isna().sum()
mgl_des_col = mgl_des_na[mgl_des_na == 0].index
mgl_descriptor = mgl_descriptor[mgl_des_col]

mgl_des_uniq = mgl_descriptor.nunique()
mgl_des_drop_idx = mgl_des_uniq[mgl_des_uniq == 1].index
mgl_descriptor.drop(mgl_des_drop_idx, axis = 1, inplace = True)


mgl_scaler = MinMaxScaler()
mgl_scaler.fit(mgl_descriptor)
mgl_scaler.data_min_
mgl_scaler.data_max_


mgl_scaled_descriptor = pd.DataFrame(mgl_scaler.transform(mgl_descriptor))
mgl_scaled_descriptor.columns = mgl_descriptor.columns
round(mgl_scaled_descriptor, 5)

mgl_x = pd.concat([mgl_fingerprint, round(mgl_scaled_descriptor, 5)], axis = 1)
mgl_y_tmp = lc50_mgl['value'].drop(ms_none_idx).reset_index(drop = True)
mgl_y = pd.DataFrame({'value': mgl_y_tmp,
                     'category': pd.cut(mgl_y_tmp, bins = [0, 0.5, 2.0, 10, 20, np.infty], labels = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5'])})
mgl_y['category'].value_counts()


#%%
train_mgl_idx = random.sample(list(mgl_y.index), int(len(mgl_y) * 0.8))
test_mgl_idx = list(set(mgl_y.index) - set(train_mgl_idx))

train_mgl_x = mgl_x.iloc[train_mgl_idx]; test_mgl_x = mgl_x.iloc[test_mgl_idx]
train_mgl_y = mgl_y['category'][train_mgl_idx]; test_mgl_y = mgl_y['category'][test_mgl_idx]


train_ppm_idx = random.sample(list(ppm_y.index), int(len(ppm_y) * 0.8))
test_ppm_idx = list(set(ppm_y.index) - set(train_ppm_idx))

train_ppm_x = ppm_x.iloc[train_ppm_idx]; test_ppm_x = ppm_x.iloc[test_ppm_idx]
train_ppm_y = ppm_y['category'][train_ppm_idx]; test_ppm_y = ppm_y['category'][test_ppm_idx]


#%%
'''
    mg/L 데이터
'''
gnb = GaussianNB()
gnb.fit(train_mgl_x, train_mgl_y)
gnb_pred = gnb.predict(test_mgl_x)

# confusion_matrix(test_mgl_y, gnb_pred)
plot_confusion_matrix(gnb, test_mgl_x, test_mgl_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('GNB Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_mgl_y, gnb_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_mgl_y, gnb_pred))



#%%
logistic = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
logistic.fit(train_mgl_x, train_mgl_y)
logistic_pred = logistic.predict(test_mgl_x)

plot_confusion_matrix(logistic, test_mgl_x, test_mgl_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('Logistic Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_mgl_y, logistic_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_mgl_y, logistic_pred))


#%%
svc = SVC(C=50, kernel='rbf', gamma=1)     
svc.fit(train_mgl_x, train_mgl_y)
svc_pred = svc.predict(test_mgl_x)

plot_confusion_matrix(svc, test_mgl_x, test_mgl_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('SVM Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_mgl_y, svc_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_mgl_y, svc_pred))


#%%
rf = RandomForestClassifier()
rf.fit(train_mgl_x, train_mgl_y)
rf_pred = rf.predict(test_mgl_x)

plot_confusion_matrix(rf, test_mgl_x, test_mgl_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('RF Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_mgl_y, rf_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_mgl_y, rf_pred))

# print('training accuracy: ', rf.score(train_mgl_x, train_mgl_y))
# print('training accuracy: ', rf.score(test_mgl_x, test_mgl_y))


#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
knn.fit(train_mgl_x, train_mgl_y)
knn_pred = knn.predict(test_mgl_x)

plot_confusion_matrix(knn, test_mgl_x, test_mgl_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('KNN Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_mgl_y, knn_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_mgl_y, knn_pred))


#%%
lgb = LGBMClassifier(n_estimators=1000)
lgb.fit(train_mgl_x, train_mgl_y)
lgb_pred = lgb.predict(test_mgl_x)

plot_confusion_matrix(lgb, test_mgl_x, test_mgl_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('LightGBM Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_mgl_y, lgb_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_mgl_y, lgb_pred))


#%%
train_mgl_x = tf.cast(train_mgl_x, tf.float32)
test_mgl_x = tf.cast(test_mgl_x, tf.float32)

nn_train_y = [0 if train_mgl_y[i] == 'cat1' 
              else 1 if train_mgl_y[i] == 'cat2' 
              else 2 if train_mgl_y[i] == 'cat3' 
              else 3 if train_mgl_y[i] == 'cat4' 
              else 4 for i in train_mgl_y.index]
nn_train_y = tf.cast(nn_train_y, tf.int32)

nn_test_y = [0 if test_mgl_y[i] == 'cat1' 
             else 1 if test_mgl_y[i] == 'cat2' 
             else 2 if test_mgl_y[i] == 'cat3' 
             else 3 if test_mgl_y[i] == 'cat4' 
             else 4 for i in test_mgl_y.index]
nn_test_y = tf.cast(nn_test_y, tf.int32)


#%%
input1 = layers.Input((train_mgl_x.shape[1]))

dense1 = layers.Dense(100, activation = 'relu')
dense2 = layers.Dense(50, activation = 'tanh')
dense3 = layers.Dense(5, activation = 'softmax')

yhat = dense3(dense2(dense1(input1)))

model = K.models.Model(input1, yhat)
model.summary()


#%%
adam = K.optimizers.Adam(0.001)
# mae = K.losses.MeanAbsoluteError()
scc = K.losses.SparseCategoricalCrossentropy()

# model.compile(optimizer = adam, loss = bc, metrics = ['accuracy'])
model.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
result = model.fit(train_mgl_x, nn_train_y, epochs = 1000, batch_size = len(nn_train_y), verbose = 1)
# result = model.fit(x_train_smote, half_train_smote, epochs = 500, batch_size = len(half_train_smote), verbose = 1)


#%%
nn_pred_prob = model.predict(test_mgl_x)
print(scc(nn_test_y, nn_pred_prob).numpy())

nn_pred = np.argmax(nn_pred_prob, axis = 1)

print('\n', classification_report(nn_test_y, nn_pred))


#%%
'''
    ppm 데이터
'''
gnb = GaussianNB()
gnb.fit(train_ppm_x, train_ppm_y)
gnb_pred = gnb.predict(test_ppm_x)

# confusion_matrix(test_mgl_y, gnb_pred)
plot_confusion_matrix(gnb, test_ppm_x, test_ppm_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('GNB Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_ppm_y, gnb_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_ppm_y, gnb_pred))



#%%
logistic = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
logistic.fit(train_ppm_x, train_ppm_y)
logistic_pred = logistic.predict(test_ppm_x)

plot_confusion_matrix(logistic, test_ppm_x, test_ppm_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('Logistic Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_ppm_y, logistic_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_ppm_y, logistic_pred))


#%%
svc = SVC(C=50, kernel='rbf', gamma=1)     
svc.fit(train_ppm_x, train_ppm_y)
svc_pred = svc.predict(test_ppm_x)

plot_confusion_matrix(svc, test_ppm_x, test_ppm_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('SVM Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_ppm_y, svc_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_ppm_y, svc_pred))


#%%
rf = RandomForestClassifier()
rf.fit(train_ppm_x, train_ppm_y)
rf_pred = rf.predict(test_ppm_x)

plot_confusion_matrix(rf, test_ppm_x, test_ppm_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('RF Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_ppm_y, rf_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_ppm_y, rf_pred))

# print('training accuracy: ', rf.score(train_mgl_x, train_mgl_y))
# print('training accuracy: ', rf.score(test_mgl_x, test_mgl_y))


#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
knn.fit(train_ppm_x, train_ppm_y)
knn_pred = knn.predict(test_ppm_x)

plot_confusion_matrix(knn, test_ppm_x, test_ppm_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('KNN Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_ppm_y, knn_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_ppm_y, knn_pred))


#%%
lgb = LGBMClassifier(n_estimators=1000)
lgb.fit(train_ppm_x, train_ppm_y)
lgb_pred = lgb.predict(test_ppm_x)

plot_confusion_matrix(lgb, test_ppm_x, test_ppm_y, normalize = 'true', cmap = plt.cm.Blues)
plt.title('LightGBM Confusion Matrix')
plt.show()
plt.close()
# print('\nAccuracy: ', accuracy_score(test_mgl_y, gnb_pred))
print('\n', classification_report(test_ppm_y, lgb_pred))
print('\nCohean Kappa Score: ', cohen_kappa_score(test_ppm_y, lgb_pred))


#%%
train_mgl_x = tf.cast(train_mgl_x, tf.float32)
test_mgl_x = tf.cast(test_mgl_x, tf.float32)

nn_train_y = [0 if train_mgl_y[i] == 'cat1' 
              else 1 if train_mgl_y[i] == 'cat2' 
              else 2 if train_mgl_y[i] == 'cat3' 
              else 3 if train_mgl_y[i] == 'cat4' 
              else 4 for i in train_mgl_y.index]
nn_train_y = tf.cast(nn_train_y, tf.int32)

nn_test_y = [0 if test_mgl_y[i] == 'cat1' 
             else 1 if test_mgl_y[i] == 'cat2' 
             else 2 if test_mgl_y[i] == 'cat3' 
             else 3 if test_mgl_y[i] == 'cat4' 
             else 4 for i in test_mgl_y.index]
nn_test_y = tf.cast(nn_test_y, tf.int32)


#%%
input1 = layers.Input((train_mgl_x.shape[1]))

dense1 = layers.Dense(100, activation = 'relu')
dense2 = layers.Dense(50, activation = 'tanh')
dense3 = layers.Dense(5, activation = 'softmax')

yhat = dense3(dense2(dense1(input1)))

model = K.models.Model(input1, yhat)
model.summary()


#%%
adam = K.optimizers.Adam(0.001)
# mae = K.losses.MeanAbsoluteError()
scc = K.losses.SparseCategoricalCrossentropy()

# model.compile(optimizer = adam, loss = bc, metrics = ['accuracy'])
model.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
result = model.fit(train_mgl_x, nn_train_y, epochs = 1000, batch_size = len(nn_train_y), verbose = 1)
# result = model.fit(x_train_smote, half_train_smote, epochs = 500, batch_size = len(half_train_smote), verbose = 1)


#%%
nn_pred_prob = model.predict(test_mgl_x)
print(scc(nn_test_y, nn_pred_prob).numpy())

nn_pred = np.argmax(nn_pred_prob, axis = 1)

print('\n', classification_report(nn_test_y, nn_pred))


