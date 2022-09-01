import sys
sys.path.append('../')

import pickle
import openpyxl
import sklearn

import numpy as np
import pandas as pd

from utils.smiles2fing import Smiles2Fing
from sklearn.cross_decomposition import PLSRegression


vapour = pd.read_excel('../data/vapour.xlsx')
aerosol = pd.read_excel('../data/aerosol.xlsx')
gas = pd.read_excel('../data/gas.xlsx')

pred_df_tmp = pd.read_excel('pred_data.xlsx')

na_idx1 = pred_df_tmp[pred_df_tmp.SMILES.isna()].index
na_idx2 = pred_df_tmp[pred_df_tmp.SMILES.isin([' '])].index

pred_df = pred_df_tmp.drop(na_idx1.append(na_idx2)).reset_index(drop = True)


def check_cas_in_train(df):
    num_ = pred_df.CasRN.isin(df.CasRN).sum()
    cas_in_train = pred_df[pred_df.CasRN.isin(df.CasRN)]
    
    return num_, cas_in_train

n, v = check_cas_in_train(gas)


def prediction(inhale_type, model):
    na_idx, x = Smiles2Fing(pred_df.SMILES)
    x.insert(0, 'time', 4)
    
    with open('../results/saved_model/' + inhale_type + '_' + model + '.pkl', 'rb') as file:
        clf = pickle.load(file)
    
    if type(clf) == sklearn.cross_decomposition._pls.PLSRegression:
        pred = np.argmax(clf.predict(x), axis = 1)
    
    else:
        pred = clf.predict(x)
    
    df_ = pd.concat([pred_df.drop(na_idx).reset_index(drop = True), 
                     pd.DataFrame({'pred': pred})],
                    axis =1)
    
    return df_


tmp = pd.merge(pred_df_tmp, prediction('gas', 'mlp')[['CasRN', 'pred']], how = 'left', on = ('CasRN'))
tmp.to_excel('tmp.xlsx', header = True, index = False)
