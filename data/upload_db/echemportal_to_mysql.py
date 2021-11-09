#%5
# !pip install pymysql
# !pip install mysql-connector-python
import os 
import pymysql
import openpyxl
import pandas as pd
from sqlalchemy import create_engine

#%%
mydb = pymysql.connect(host = '127.0.0.1', user = 'root', password = 'nrz5oloF71', 
                       cursorclass = pymysql.cursors.DictCursor, db = 'echemportal')
conn = mydb.cursor()

conn.execute('show tables;')
conn.fetchall()


#%%
fold_dir = 'C:/Users/SOYOUNG/Desktop/toxic/data/oecd_echemportal/'
fold_name = os.listdir(fold_dir + 'Preprocessed data/')
fold_name = [i for i in fold_name if i]
fold_path = [fold_dir + 'Preprocessed data/' + i for i in fold_name]

#%%
engine = create_engine("mysql+pymysql://root:nrz5oloF71@localhost/echemportal", encoding='utf8')
conn = engine.connect()

for i in range(len(fold_path)):
    data_tmp = pd.read_excel(fold_path[i], sheet_name = 'Sheet1')
    data_tmp.to_sql(name = fold_name[i].split('.')[0], con = engine, if_exists = 'replace', index = False)


#%%
fold_dir = 'C:/Users/SOYOUNG/Desktop/toxic/data/oecd_echemportal/'
fold_name = os.listdir(fold_dir + 'Data tmp/')
fold_name = [i for i in fold_name if i]
fold_path = [fold_dir + 'Data tmp/' + i for i in fold_name]


for i in range(len(fold_path)):
    data_tmp = pd.read_excel(fold_path[i], sheet_name = 'Sheet1')
    data_tmp.to_sql(name = fold_name[i].split('.')[0], con = engine, if_exists = 'replace', index = False)

