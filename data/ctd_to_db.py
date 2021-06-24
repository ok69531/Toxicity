#%%
import os 
import glob
import gzip
import zipfile
import pandas as pd
import time

import pymysql
import sqlalchemy
# pymysql.install_as_MySQLdb()
import MySQLdb

from sqlalchemy import create_engine

#%% 
# connection & create pyctd 

# mydb = pymysql.connect(host = '127.0.0.1', user = 'root', password = 'nrz5oloF71',
#                        charset = 'utf8mb4', cursorclass = pymysql.cursors.DictCursor)
# conn = mydb.cursor()
# sql_Statement = 'CREATE DATABASE pyctd' 
# conn.execute(sql_Statement)
# conn.fetchall()

engine = create_engine("mysql+mysqldb://root:nrz5oloF71@localhost/pyctd", encoding='utf8')
conn = engine.connect()
# conn.execute('CREATE DATABASE pyctd' )



#%%
fold_dir = 'C:/Users/SOYOUNG/Desktop/toxic/ctd'
fold_name = os.listdir(fold_dir)
fold_name = [i for i in fold_name if i not in ['.Rhistory', 'scrap.R', 'exposures']]
fold_path = [fold_dir + '/' + i for i in fold_name]

#%%
for i in range(len(fold_path)):
    dir_tmp = fold_path[i]
    table_name = fold_name[i]
    
    dir = [dir_tmp + '/' + os.listdir(dir_tmp)[j] for j in range(len(os.listdir(dir_tmp))) 
           if os.listdir(dir_tmp)[j][-7:] == '.csv.gz']
    data = pd.read_csv(dir[0], compression = 'gzip', header = 27, sep = ',',
                       quotechar = '"').drop(0)
    data.columns = [data.columns[0][2:]] + list(data.columns[1:])
    
    data.to_sql(name = table_name, con = engine, if_exists = 'replace', index = False)



#%% 저장되지 않은 table 확인
mydb = pymysql.connect(host = '127.0.0.1', user = 'root', password = 'nrz5oloF71',
                       charset = 'utf8mb4', cursorclass = pymysql.cursors.DictCursor, db = 'pyctd')
conn = mydb.cursor()

conn.execute('SHOW tables;')
result = conn.fetchall()
saved = [result[i]['Tables_in_pyctd'] for i in range(len(result))]
# for i in range(len(result)):
#     print(result[i]['Tables_in_pyctd'])

unsaved = list(set(fold_name) - set(saved))



#%%
unsaved_name = sorted([i for i in unsaved if i not in ['gd', 'allgenes', 'gcixntypes']])
unsaved_path = sorted([fold_path[i] for i in range(len(fold_path)) if fold_path[i].split('/')[-1] in unsaved_name])

for i in range(len(unsaved_name)):
    dir_tmp = unsaved_path[i]
    table_name = unsaved_name[i]
    
    dir = [dir_tmp + '/' + os.listdir(dir_tmp)[j] for j in range(len(os.listdir(dir_tmp))) 
           if os.listdir(dir_tmp)[j][-7:] == '.csv.gz']
    data = pd.read_csv(dir[0], compression = 'gzip', header = 26, sep = ',',
                       quotechar = '"').drop(0)
    data.columns = [data.columns[0][2:]] + list(data.columns[1:])
    
    data.to_sql(name = table_name, con = engine, if_exists = 'replace', index = False)


#%%
table_name = 'gcixntypes'
dir_tmp = [fold_path[i] for i in range(len(fold_path)) if fold_path[i].split('/')[-1] == 'gcixntypes'][0]

dir = [dir_tmp + '/' + os.listdir(dir_tmp)[j] for j in range(len(os.listdir(dir_tmp))) 
        if os.listdir(dir_tmp)[j][-4:] == '.csv']
data = pd.read_csv(dir[0], header = 24, sep = ',', quotechar = '"').drop(0)
data.columns = [data.columns[0][2:]] + list(data.columns[1:])

data.to_sql(name = table_name, con = engine, if_exists = 'replace', index = False)



#%% save allgenes
dir_tmp = fold_path[3]
table_name = fold_name[3]

dir = [dir_tmp + '/' + os.listdir(dir_tmp)[j] for j in range(len(os.listdir(dir_tmp))) 
        if os.listdir(dir_tmp)[j][-7:] == '.csv.gz']
data = pd.read_csv(dir[0], compression = 'gzip', header = 27, sep = ',',
                    quotechar = '"').drop(0)
data.columns = [data.columns[0][2:]] + list(data.columns[1:])

data.to_sql(name = table_name, con = engine, if_exists = 'replace', index = False,
                dtype = {
                    'AltGeneIDs': sqlalchemy.types.TEXT(20000), 
                    })



#%% save gene-disease
gd_dir = fold_path[-3] + '/' + os.listdir(fold_path[-3])[0]
zf = zipfile.ZipFile(gd_dir)

s1 = time.time()
gd = pd.read_csv(zf.open(zf.namelist()[0]), header = 27, sep = ',', quotechar = '"').drop(0)
gd.columns = [gd.columns[0][2:]] + list(gd.columns[1:])
print('time : ', time.time() - s1)

gd.shape

s2 = time.time()
gd.to_sql(name = 'gd', con = engine, chunksize = 10000, 
          if_exists = 'append', index = False)
# gd.to_sql(name = 'gd', con = engine, chunksize = 10000, 
#           if_exists = 'append', index = False, method = 'multi')
print('time : ', time.time() - s2)

# 3993.398815393448 / 60
# 한시간걸림


#%%
# connection.close()
