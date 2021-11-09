#%%
import time
import random
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import sklearn
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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


#%%
path = 'C:/Users/SOYOUNG/Desktop/toxic/data/'

genotoxicity_tmp = pd.read_excel(path + 'genotoxicity/genotoxicity_input.xlsx')

fingerprints_tmp = genotoxicity_tmp.filter(regex = 'fingerprint|Fingerprint|(Bin)')

fingerprints_tmp.columns[-3]
len(list(fingerprints_tmp.iloc[0, -3])) # EState fingerprints for SMILES (Bin)


fingerprints_tmp['EState fingerprints for SMILES (Bin)'].isna().sum()
estate_fingerprint = np.array([])
estate_fingerprint = [np.append(estate_fingerprint, list(i), axis = 0) for i in 
                     tqdm(fingerprints_tmp['EState fingerprints for SMILES (Bin)'].values)]
estate_fingerprint = pd.DataFrame(estate_fingerprint).astype(int)
estate_fingerprint.dtypes.unique()


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


#%%
genotoxicity_tmp.columns[:5]
geno_con = genotoxicity_tmp['Genotoxicity(진보)']
geno_pro = genotoxicity_tmp['Genotoxicity(보수)']
geno_half = genotoxicity_tmp['Genotoxicity(0.5기준)']

# geno_pro.value_counts()
# geno_con.value_counts()
# geno_half.value_counts()

# plt.hist(geno_pro); plt.show()
# plt.hist(geno_con); plt.show()
# plt.hist(geno_half); plt.show()


geno_pro.value_counts()
geno_pro.value_counts(normalize = True) * 100
geno_pro.value_counts(normalize = True).plot(kind = 'bar'); plt.show()
geno_con.value_counts(normalize = True) * 100
geno_con.value_counts(normalize = True).plot(kind = 'bar'); plt.show()
geno_half.value_counts(normalize = True) * 100
geno_half.value_counts(normalize = True).plot(kind = 'bar'); plt.show()


pro = pd.Series([0 if geno_pro[i] == 'N' else 1 for i in range(genotoxicity_tmp.shape[0])])
con = pd.Series([0 if geno_con[i] == 'N' else 1 for i in range(genotoxicity_tmp.shape[0])])
half = pd.Series([0 if geno_half[i] == 'N' else 1 for i in range(genotoxicity_tmp.shape[0])])



#%%
data = pd.concat([scaled_descriptor, estate_fingerprint], axis = 1)


train_idx = random.sample(range(data.shape[0]), round(data.shape[0] * 0.7))
test_idx = list(set(range(data.shape[0])) - set(train_idx))


x_train = data.iloc[train_idx]; x_test = data.iloc[test_idx]
# esate_train = estate_fingerprint.iloc[train_idx]; etate_test = estate_fingerprint.iloc[test_idx]
# des_train = descriptor.iloc[train_idx]; des_test = descriptor.iloc[test_idx]


pro_train = pro[train_idx]; pro_test = pro[test_idx]
con_train = con[train_idx]; con_test = con[test_idx]
half_train = half[train_idx]; half_test = half[test_idx]


# pro.value_counts()[0]/len(pro)
# pro_train.value_counts()[0]/len(pro_train)
# con.value_counts()[0]/len(con)
# con_train.value_counts()[0]/len(con_train)
# half.value_counts()[0]/len(half)
# half_train.value_counts()[0]/len(half_train)

pro.value_counts()
pro_train.value_counts(normalize = True) * 100
pro_train.value_counts(normalize = True).plot(kind = 'bar'); plt.show()
con_train.value_counts(normalize = True) * 100
con_train.value_counts(normalize = True).plot(kind = 'bar'); plt.show()
half_train.value_counts(normalize = True) * 100
half_train.value_counts(normalize = True).plot(kind = 'bar'); plt.show()


#%%
''' 
    진보적인 방법 (하나라도 P이면 P) 
    0: Negative
    1: Positive
'''

gnb = GaussianNB()
gnb.fit(x_train, pro_train)
gnb_pred = gnb.predict(x_test)

# confusion_matrix(pro_test, gnb_pred, labels = [0, 1])
print('\nPrecision: ', precision_score(pro_test, gnb_pred))
print('\nRecall: ', recall_score(pro_test, gnb_pred))
print('\nF1 score: ', f1_score(pro_test, gnb_pred))
print('\nAUC: ', roc_auc_score(pro_test, gnb_pred))
print('\nAccuracy: ', accuracy_score(pro_test, gnb_pred))
print('\n', classification_report(pro_test, gnb_pred))

confusion_matrix(pro_test, gnb_pred)
plot_confusion_matrix(gnb, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('GNB Confusion Matrix')
plt.show()
plt.close()


#%%
logistic = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
logistic.fit(x_train, pro_train)
logistic_pred = logistic.predict(x_test)

print('\nPrecision: ', precision_score(pro_test, logistic_pred))
print('\nRecall: ', recall_score(pro_test, logistic_pred))
print('\nF1 score: ', f1_score(pro_test, logistic_pred))
print('\nAUC: ', roc_auc_score(pro_test, logistic_pred))
print('\nAccuracy: ', accuracy_score(pro_test, logistic_pred))
print('\n', classification_report(pro_test, logistic_pred))

confusion_matrix(pro_test, logistic_pred)
plot_confusion_matrix(logistic, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('Logistic Confusion Matrix')
plt.show()
plt.close()


#%%
svc = SVC(C=50, kernel='rbf', gamma=1)     
svc.fit(x_train, pro_train)
svc_pred = svc.predict(x_test)

print('\nPrecision: ', precision_score(pro_test, svc_pred))
print('\nRecall: ', recall_score(pro_test, svc_pred))
print('\nF1 score: ', f1_score(pro_test, svc_pred))
print('\nAUC: ', roc_auc_score(pro_test, svc_pred))
print('\nAccuracy: ', accuracy_score(pro_test, svc_pred))
print('\n', classification_report(pro_test, svc_pred))

plot_confusion_matrix(svc, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('SVM Confusion Matrix')
plt.show()
plt.close()


#%%
rf = RandomForestClassifier()
rf.fit(x_train, pro_train)
rf_pred = rf.predict(x_test)

print('\nPrecision: ', precision_score(pro_test, rf_pred))
print('\nRecall: ', recall_score(pro_test, rf_pred))
print('\nF1 score: ', f1_score(pro_test, rf_pred))
print('\nAUC: ', roc_auc_score(pro_test, rf_pred))
print('\nAccuracy: ', accuracy_score(pro_test, rf_pred))
print('\n', classification_report(pro_test, rf_pred))

plot_confusion_matrix(rf, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('RF Confusion Matrix')
plt.show()
plt.close()


#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree')
knn.fit(x_train, pro_train)
knn_pred = knn.predict(x_test)

print('\nPrecision: ', precision_score(pro_test, knn_pred))
print('\nRecall: ', recall_score(pro_test, knn_pred))
print('\nF1 score: ', f1_score(pro_test, knn_pred))
print('\nAUC: ', roc_auc_score(pro_test, knn_pred))
print('\nAccuracy: ', accuracy_score(pro_test, knn_pred))
print('\n', classification_report(pro_test, knn_pred))

plot_confusion_matrix(knn, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('KNN Confusion Matrix')
plt.show()
plt.close()


#%%
lgb = LGBMClassifier(n_estimators = 1000)
lgb.fit(x_train, pro_train)
lgb_pred = lgb.predict(x_test)

print('\nPrecision: ', precision_score(pro_test, lgb_pred))
print('\nRecall: ', recall_score(pro_test, lgb_pred))
print('\nF1 score: ', f1_score(pro_test, lgb_pred))
print('\nAUC: ', roc_auc_score(pro_test, lgb_pred))
print('\nAccuracy: ', accuracy_score(pro_test, lgb_pred))
print('\n', classification_report(pro_test, lgb_pred))

plot_confusion_matrix(lgb, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('LightGBM Confusion Matrix')
plt.show()
plt.close()


#%%
'''
    MLP (activation: softmax) classification
'''
#%%
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

pro_train = np.array(pro_train).astype(int)
pro_test = np.array(pro_test).astype(int)


#%%
input1 = layers.Input((x_train.shape[1]))

dense1 = layers.Dense(30, activation = 'relu')
dense2 = layers.Dense(10, activation = 'tanh')
dense3 = layers.Dense(2, activation = 'softmax')

yhat = dense3(dense2(dense1(input1)))

model = K.models.Model(input1, yhat)
model.summary()


#%%
adam = K.optimizers.Adam(0.001)
# mae = K.losses.MeanAbsoluteError()
scc = K.losses.SparseCategoricalCrossentropy()

# model.compile(optimizer = adam, loss = bc, metrics = ['accuracy'])
model.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
result = model.fit(x_train, pro_train, epochs = 1000, batch_size = len(pro_train), verbose = 1)
# result = model.fit(x_train_smote, half_train_smote, epochs = 500, batch_size = len(half_train_smote), verbose = 1)


#%%
nn_pred_prob = model.predict(x_test)
print(scc(pro_test, nn_pred_prob).numpy())

nn_pred = np.argmax(nn_pred_prob, axis = 1)

# pd.crosstab(np.argmax(nn_pred, axis = 1), pro_test)
# roc_auc_score(pro_test, np.argmax(nn_pred, axis = 1))


print('\nPrecision: ', precision_score(pro_test, nn_pred))
print('\nRecall: ', recall_score(pro_test, nn_pred))
print('\nF1 score: ', f1_score(pro_test, nn_pred))
print('\nAUC: ', roc_auc_score(pro_test, nn_pred))
print('\nAccuracy: ', accuracy_score(pro_test, nn_pred))
print('\n', classification_report(pro_test, nn_pred))


#%%
# !pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = 0)
x_train_smote, pro_train_smote = smote.fit_resample(x_train, pro_train)
print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', x_train.shape, pro_train.shape)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', x_train_smote.shape, pro_train_smote.shape)
print('SMOTE 적용 후 레이블 값 분포:')
print(pd.Series(pro_train_smote).value_counts())


#%%
gnb = GaussianNB()
gnb.fit(x_train_smote, pro_train_smote)
gnb_pred = gnb.predict(x_test)

# confusion_matrix(pro_test, gnb_pred, labels = [0, 1])
print('\nPrecision: ', precision_score(pro_test, gnb_pred))
print('\nRecall: ', recall_score(pro_test, gnb_pred))
print('\nF1 score: ', f1_score(pro_test, gnb_pred))
print('\nAUC: ', roc_auc_score(pro_test, gnb_pred))
print('\nAccuracy: ', accuracy_score(pro_test, gnb_pred))
print('\n', classification_report(pro_test, gnb_pred))

plot_confusion_matrix(gnb, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('GNB Confusion Matrix')
plt.show()
plt.close()


#%%
logistic = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
logistic.fit(x_train_smote, pro_train_smote)
logistic_pred = logistic.predict(x_test)

print('\nPrecision: ', precision_score(pro_test, logistic_pred))
print('\nRecall: ', recall_score(pro_test, logistic_pred))
print('\nF1 score: ', f1_score(pro_test, logistic_pred))
print('\nAUC: ', roc_auc_score(pro_test, logistic_pred))
print('\nAccuracy: ', accuracy_score(pro_test, logistic_pred))
print('\n', classification_report(pro_test, logistic_pred))

plot_confusion_matrix(logistic, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('Logistic Confusion Matrix')
plt.show()
plt.close()


#%%
svc = SVC(C=50, kernel='rbf', gamma=1)     
svc.fit(x_train_smote, pro_train_smote)
svc_pred = svc.predict(x_test)

print('\nPrecision: ', precision_score(pro_test, svc_pred))
print('\nRecall: ', recall_score(pro_test, svc_pred))
print('\nF1 score: ', f1_score(pro_test, svc_pred))
print('\nAUC: ', roc_auc_score(pro_test, svc_pred))
print('\nAccuracy: ', accuracy_score(pro_test, svc_pred))
print('\n', classification_report(pro_test, svc_pred))

plot_confusion_matrix(svc, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('SVM Confusion Matrix')
plt.show()
plt.close()


#%%
rf = RandomForestClassifier()
rf.fit(x_train_smote, pro_train_smote)
rf_pred = rf.predict(x_test)

print('\nPrecision: ', precision_score(pro_test, rf_pred))
print('\nRecall: ', recall_score(pro_test, rf_pred))
print('\nF1 score: ', f1_score(pro_test, rf_pred))
print('\nAUC: ', roc_auc_score(pro_test, rf_pred))
print('\nAccuracy: ', accuracy_score(pro_test, rf_pred))
print('\n', classification_report(pro_test, rf_pred))

plot_confusion_matrix(rf, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('RF Confusion Matrix')
plt.show()
plt.close()


#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree')
knn.fit(x_train_smote, pro_train_smote)
knn_pred = knn.predict(x_test)

print('\nPrecision: ', precision_score(pro_test, knn_pred))
print('\nRecall: ', recall_score(pro_test, knn_pred))
print('\nF1 score: ', f1_score(pro_test, knn_pred))
print('\nAUC: ', roc_auc_score(pro_test, knn_pred))
print('\nAccuracy: ', accuracy_score(pro_test, knn_pred))
print('\n', classification_report(pro_test, knn_pred))

plot_confusion_matrix(knn, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('KNN Confusion Matrix')
plt.show()
plt.close()


#%%
lgb = LGBMClassifier(n_estimators=1000)
lgb.fit(x_train_smote, pro_train_smote)
lgb_pred = lgb.predict(x_test)

print('\nPrecision: ', precision_score(pro_test, lgb_pred))
print('\nRecall: ', recall_score(pro_test, lgb_pred))
print('\nF1 score: ', f1_score(pro_test, lgb_pred))
print('\nAUC: ', roc_auc_score(pro_test, lgb_pred))
print('\nAccuracy: ', accuracy_score(pro_test, lgb_pred))
print('\n', classification_report(pro_test, lgb_pred))

plot_confusion_matrix(lgb, x_test, pro_test, normalize = 'true', cmap = plt.cm.Blues)
plt.title('LightGBM Confusion Matrix')
plt.show()
plt.close()


#%%
'''
    MLP (activation: softmax) classification
'''
#%%
x_train_smote = tf.cast(x_train_smote, tf.float32)
x_test = tf.cast(x_test, tf.float32)

pro_train_smote = np.array(pro_train_smote).astype(int)
pro_test = np.array(pro_test).astype(int)


#%%
input1 = layers.Input((x_train.shape[1]))

dense1 = layers.Dense(30, activation = 'relu')
dense2 = layers.Dense(10, activation = 'tanh')
dense3 = layers.Dense(2, activation = 'softmax')

yhat = dense3(dense2(dense1(input1)))

model = K.models.Model(input1, yhat)
model.summary()


#%%
adam = K.optimizers.Adam(0.001)
# mae = K.losses.MeanAbsoluteError()
scc = K.losses.SparseCategoricalCrossentropy()

# model.compile(optimizer = adam, loss = bc, metrics = ['accuracy'])
model.compile(optimizer = adam, loss = scc, metrics = ['accuracy'])
result = model.fit(x_train, pro_train, epochs = 1000, batch_size = len(pro_train), verbose = 1)
# result = model.fit(x_train_smote, half_train_smote, epochs = 500, batch_size = len(half_train_smote), verbose = 1)


#%%
nn_pred_prob = model.predict(x_test)
print(scc(pro_test, nn_pred_prob).numpy())

nn_pred = np.argmax(nn_pred_prob, axis = 1)

pd.crosstab(nn_pred, pro_test)
roc_auc_score(pro_test, nn_pred)


print('\nPrecision: ', precision_score(pro_test, nn_pred))
print('\nRecall: ', recall_score(pro_test, nn_pred))
print('\nF1 score: ', f1_score(pro_test, nn_pred))
print('\nAUC: ', roc_auc_score(pro_test, nn_pred))
print('\nAccuracy: ', accuracy_score(pro_test, nn_pred))
print('\n', classification_report(pro_test, nn_pred))



