import random
import logging
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, truncexpon, genpareto
from typing import Dict, Any
import os
from numpy.random import choice
import argparse
from tqdm import tqdm
from physical_db import DBConnection, TrueCardinalityEstimator
import time
import psycopg2
from sqlalchemy import create_engine
import pickle
import re
import stats as sts
from sklearn import metrics
import seaborn as sns


# Input: num_pnode, dict_node, A  Output: Null(modified dict_node)
def Pnode(num_pnode, dict_node, A):
    for pn in range(num_pnode):  # 
        df = pd.DataFrame()
        pkrows = random.randint(1e4, 5e4)
        for pkr in range(pkrows):
            df = df.append(pd.DataFrame({'col0':[pkr]}), ignore_index=True)
        dict_node[pn] = df



# Input: num_fnode, num_pnode, A, dict_node   Output: Null(modified dict_node)
def Fnode(num_fnode, num_pnode, dict_node, A, W):
    if num_pnode-1 == 0:
        for fn in range(num_fnode):  
            df = pd.DataFrame()
            fkrows = random.randint(1e4, 5e4)
            
            join_sele = random.uniform(0.3,0.9)
            listpk1 = dict_node[0]['col0'].to_frame().sample(frac = join_sele)['col0'].to_list()  
            for fkr in range(fkrows):
                df = df.append(pd.DataFrame({'col0':[random.choice(listpk1)]}), ignore_index=True)
            dict_node[num_pnode+fn] = df
            A[num_pnode+fn][0] = 1
            W[num_pnode+fn][0] = join_sele

    else:
        ex2fk = 1 # random.randint(0,1)    # modify
        if(ex2fk == 1):
            df = pd.DataFrame()
            fkrows = random.randint(1e4, 5e4)
            join_sele = random.uniform(0.3,0.9)
            listpk1 = dict_node[0]['col0'].to_frame().sample(frac = join_sele)['col0'].to_list() 
            for fkr in range(fkrows):
                df = df.append(pd.DataFrame({'col0':[random.choice(listpk1)]}), ignore_index=True)
            A[num_pnode][0] = 1
            W[num_pnode][0] = join_sele
            join_sele = random.uniform(0.3,0.9)
            listpk2 = dict_node[1]['col0'].to_frame().sample(frac = join_sele)['col0'].to_list()  
            tmp = []
            for fkr in range(df.shape[0]):
                tmp.append(random.choice(listpk2))
            df['col1'] = tmp
            A[num_pnode][1] = 1
            W[num_pnode][1] = join_sele
            dict_node[num_pnode] = df

        for fn in range(ex2fk, num_fnode):
            df = pd.DataFrame()
            fkrows = random.randint(1e4, 5e4)
            s_pn = random.randint(1, 2)  
            join_sele = random.uniform(0.3,0.9)
            listpk = dict_node[s_pn-1][f'col0'].to_frame().sample(frac = join_sele)[f'col0'].to_list()  
            for fkr in range(fkrows):
                df = df.append(pd.DataFrame({'col0':[random.choice(listpk)]}), ignore_index=True)
            dict_node[num_pnode+fn] = df
            A[num_pnode+fn][s_pn-1] = 1
            W[num_pnode+fn][s_pn-1] = join_sele
    


# Input: num_fnode, num_pnode, dict_node   Output: Null(modified dict_node)
def Add_cols(num_fnode, num_pnode, dict_node):
    dom_list = list(range(50, 800, 1))
    corr_list = list(range(0, 3, 1))
    skew_list = list(range(1, 10, 1))

    for tn in tqdm(range(num_fnode + num_pnode)):  
        df = dict_node[tn]
        col_num = [2,2,2,2,3]
        cols = choice(col_num)  # 2 # random.randint(2,3)  # modify
        seed = random.randint(0,9)
        ist = df.shape[1]

        for i in range(ist, ist+cols):
            # print(i == df.shape[1])
            seed = seed + 1
            row_num = len(df) # df.shape[0]  
            dom = choice(dom_list)
            corr = choice(corr_list)
            skew = choice(skew_list)
            corr = corr/10
            skew = skew/10  # for create table

            random.seed(seed)
            np.random.seed(seed)

            if i == ist:
                # generate the first column according to skew
                col0 = np.arange(dom) # make sure every domain value has at least 1 value
                tmp = genpareto.rvs(skew-1, size=row_num-len(col0)) # c = skew - 1, so we can have c >= 0
                tmp = ((tmp - tmp.min()) / (tmp.max() - tmp.min())) * dom # rescale generated data to the range of domain
                col0 = np.concatenate((col0, np.clip(tmp.astype(int), 0, dom-1)))
                df[f'col{i}'] = col0
                continue

            else: 
                # generate the others
                col_ad = []
                tmp = genpareto.rvs(skew-1, size=row_num) # c = skew - 1, so we can have c >= 0
                tmp = ((tmp - tmp.min()) / (tmp.max() - tmp.min())) * dom # rescale generated data to the range of domain
                tmp = tmp.astype(int)
                for v in col0:
                    # print(np.random.uniform(0, 1) <= corr)
                    col_ad.append(v if (np.random.uniform(0, 1) <= corr and v <= dom) else np.random.choice(tmp))
                df[f'col{i}'] = col_ad
                col0 = col_ad  
            dict_node[tn] = df


# export Schema_sql
# Input: num_fnode, num_pnode, di, dict_node  Output: schema_sql
def Get_schema(num_fnode, num_pnode, di, dict_node):
    schema_sql = ''
    for ti in range(num_pnode+num_fnode):
        df_t = dict_node[ti]
        t_head = list(df_t.columns)
        schema_sql += f'DROP TABLE IF EXISTS table{di}_{ti};\n'
        schema_sql += 'CREATE TABLE ' + f'table{di}_{ti}' + '(\n'
        for item in t_head:
            schema_sql += '    ' + item + ' integer NOT NULL,\n'
        schema_sql = schema_sql[0: len(schema_sql)-2] + '\n);\n'
    return schema_sql
    # print(schema_sql)


# imort to database 
# Input: schema_sql, num_fnode, num_pnode, di, dict_node   Output: Null(modified database)
def table2pg(schema_sql, num_fnode, num_pnode, di, dict_node):
    db_connection = DBConnection(db='card_advisor', db_password="jintao2020", db_user='jintao', db_host="localhost")  # modify
    db_connection.submit_query(schema_sql)
    engine = create_engine('postgresql+psycopg2://jintao:jintao2020@localhost:5432/card_advisor')
    for ti in range(num_pnode+num_fnode):
        dict_node[ti].to_sql(f'table{di}_{ti}', engine, if_exists='replace', index=False)


# generate SQL and get truecard  
# Input: di, A, dict_node, sql_num, num_pnode   Output: (sql_truecard to file); generate time to plk
def Sql_truecard(di, A , dict_node, sql_num, num_pnode):
    ftrain = open('./benchmark/' + f'{di}_multi' + 'train.sql','w')
    ftest = open('./benchmark/' + f'{di}_multi' + 'test.sql','w')
    db_connection = DBConnection(db='card_advisor', db_password="jintao2020", db_user='jintao', db_host="localhost")  # modify

    ops = ['=', '<', '>', '<', '>']  
    sql_count = 0
    generate_st = time.time()
    while(1):
        sqli = 'SELECT COUNT(*) FROM '
        for table in range(len(dict_node)):
            sqli += f'table{di}_{table}, '
        sqli = sqli[0:-2] + ' WHERE '

        '''
        if num_pnode != 1:
            sqli += f'table{di}_{1}.col1=table{di}_{0}.col0 AND '
        '''

        [rows, cols] = A.shape
        for i in range(num_pnode,rows):
            if list(A[i]).count(1)==1:
                sqli += f'table{di}_{i}.col0=table{di}_{list(A[i]).index(1)}.col0 AND '
            elif list(A[i]).count(1)==2:
                sqli += f'table{di}_{i}.col0=table{di}_{0}.col0 AND '
                sqli += f'table{di}_{i}.col1=table{di}_{1}.col0 AND '
        sqli += '  '
        for table in range(len(dict_node)):
            columns = list(dict_node[table].columns)

            col = list(columns[1+list(A[table]).count(1):])[0]
            sqli += f'table{di}_{table}.{col}' + choice(ops) + str(choice(list(dict_node[table][col]))) + ' AND '
            for col in columns[2+list(A[table]).count(1):]:   # dict_node[table].columns[1:]:
                judge = random.randint(0,1)
                if judge:
                    sqli += f'table{di}_{table}.{col}' + choice(ops) + str(choice(list(dict_node[table][col]))) + ' AND '
        sqli = sqli[0:-5]
        sqli += ';'

        # print(sqli)
        true_estimator = TrueCardinalityEstimator(db_connection)
        try: 
            cardinality_true = true_estimator.true_cardinality(sqli)
            # print(sqli)
            # print(cardinality_true)
            if cardinality_true == 0:
                continue
            # sqli=sqli[0:len(sqli)-1]
            # print(sqli)
            if sql_count < sql_num:
                ftrain.write(sqli+',')
                ftrain.write(str(cardinality_true))
                ftrain.write('\n')
            else:
                ftest.write(sqli+',')
                ftest.write(str(cardinality_true))
                ftest.write('\n') 
            sql_count = sql_count+1
            if sql_count >= sql_num+200:
                break
        except Exception as e:
            pass
        continue
    ftrain.close()
    ftest.close()
    if not os.path.exists('./feature/data_generate_time.dict'):
        dict_datatime = {}
        ftime = open('./feature/data_generate_time.dict','wb')
        pickle.dump(dict_datatime,ftime)
    ftime = open('./feature/data_generate_time.dict','rb')
    dict_datatime = pickle.load(ftime)
    dict_datatime[di] = time.time()-generate_st
    ftime = open('./feature/data_generate_time.dict','wb')
    pickle.dump(dict_datatime,ftime)
    print(f'dataset{di} done')


def get_usecols(A,di,num_pnode,dict_node):
    usecols = []
    for i in range(len(dict_node)):
        if i<num_pnode:
            for col in list(dict_node[i].columns)[1:]:
                usecols.append(f'table{di}_{i}.'+col)
        else:
            for col in list(dict_node[i].columns)[np.sum(A[i] == 1):]:
                usecols.append(f'table{di}_{i}.'+col)
    return usecols


def get_join(A,di,num_pnode,dict_node):
    rows = A.shape[0]
    getdata_sql = 'SELECT * FROM '
    for table in range(len(dict_node)):
        getdata_sql += f'table{di}_{table}, '
    getdata_sql = getdata_sql[0:-2] + ' WHERE '
    for i in range(num_pnode,rows):
        if list(A[i]).count(1)==1:
            getdata_sql += f'table{di}_{i}.col0=table{di}_{list(A[i]).index(1)}.col0 AND '
        elif list(A[i]).count(1)==2:
            getdata_sql += f'table{di}_{i}.col0=table{di}_{0}.col0 AND '
            getdata_sql += f'table{di}_{i}.col1=table{di}_{1}.col0 AND '
    getdata_sql = getdata_sql[0:-5]+';'
    db_connection = DBConnection(db='card_advisor', db_password="jintao2020", db_user='jintao', db_host="localhost")
    res = db_connection.get_dataframe(getdata_sql)
    res = res.drop(['col0'],axis=1)
    new_columns = []
    i = -1
    for col in list(res.columns):
        if col=='col1':
            i+=1
        new_col = 'table'+str(di)+'_'+str(i)+col
        new_columns.append(new_col)
    res.columns = new_columns

    for row in range(rows):
        if np.sum(A[row] == 1)==2:
            # print(row)
            res = res.drop(['table'+str(di)+'_'+str(row)+'col1'],axis=1)
    
    new_columns = [f'col{i}' for i in range(len(res.columns))]
    res.columns = new_columns
    res.to_csv(f'./benchmark/table{di}.csv',index=False)
    res.to_csv(f'./benchmark/table_nohead/table{di}.csv',index=False,header=None)

    db_connection = DBConnection(db='card_advisor', db_password="jintao2020", db_user='jintao', db_host="localhost")  # modify
    engine = create_engine('postgresql+psycopg2://jintao:jintao2020@localhost:5432/card_advisor')
    res.to_sql(f'table{di}', engine, if_exists='replace', index=False)


def sql_trans(A,di,num_pnode,dict_node):
    usecols = get_usecols(A,di,num_pnode,dict_node)

    train0 = open(f'./benchmark/{di}_multitrain.sql').read()
    ftrain1 = open(f'./benchmark/table{di}train.sql', 'w')
    train1 = re.sub('FROM.*?   ', f'FROM table{di} Syn WHERE ', train0)
    for i,col in enumerate(usecols):
        train1 = re.sub(col, f'Syn.col{i}', train1)
    ftrain1.write(train1)
    ftrain1.close()

    test0 = open(f'./benchmark/{di}_multitest.sql').read()
    ftest1 = open(f'./benchmark/table{di}test.sql', 'w')
    test1 = re.sub('FROM.*?   ', f'FROM table{di} Syn WHERE ', test0)
    for i,col in enumerate(usecols):
        test1 = re.sub(col, f'Syn.col{i}', test1)
    ftest1.write(test1)
    ftest1.close()


# feature dict
def get_feature(df):
    df_feature = pd.DataFrame(columns=['col0', 'col1', 'col2', 'col3', 'col4'], index = ['skewness', 'kurtosis', 'standard_deviation', 'mean_deviation', 'range', 'distinct_values', 'corr0', 'corr1','corr2'])
    # index = ['skewness', 'kurtosis', 'standard deviation', 'mean deviation', 'range', 'coefficient of variation', 'distinct values', 'corr0', 'corr1','corr2']
    column_len = len(df.columns)
    for i in range(column_len):
        df_feature.loc['skewness', f'col{i}'] = sts.skewness(list(df.iloc[:,i]))
        df_feature.loc['kurtosis', f'col{i}'] = sts.kurtosis(list(df.iloc[:,i]))
        tmp = np.array(df.iloc[:,i])
        df_feature.loc['standard_deviation', f'col{i}'] = tmp.std()/10  
        df_feature.loc['mean_deviation', f'col{i}'] = (tmp - tmp.mean()).sum()/len(tmp)*1e15
        df_feature.loc['range', f'col{i}'] = (tmp.max()-tmp.min())/300
        df_feature.loc['distinct_values', f'col{i}'] = len(np.unique(tmp))/100
        # print(len(tmp))

        for n in range(column_len):  # col_n
            X = df.iloc[:,n] 
            for j in range(n, column_len):  # corr_j
                Y = df.iloc[:,j]
                df_feature.loc[f'corr{j}', f'col{n}'] = metrics.normalized_mutual_info_score(X, Y)  
        
        for n in range(column_len):  # col_n
            for j in range(n):  # corr_j
                df_feature.loc[f'corr{j}', f'col{n}'] = df_feature.loc[f'corr{n}', f'col{j}']
        df_feature = df_feature.fillna(0)

        array_feature = np.array(df_feature)
        # array_feature_normed = array_feature.T - array_feature.min(axis=1)/ (array_feature.max(axis=1) - array_feature.min(axis=1))
        array_feature = array_feature.reshape(45,)
        rows,cols = df.shape
        array_feature = np.append(np.array([rows/2e4,cols/2.0]),array_feature)
    return array_feature # array_feature_normed.T  # 


# Output: dict
def get_features(dict_node,A,W,di):
    dict_feature = {}
    for i in range(len(dict_node)):
        use_cols = dict_node[i].drop('col0', axis=1)
        if np.sum(A[i] == 1)==2:
            use_cols = use_cols.drop('col1', axis=1)
        # print(use_cols.columns)
        dict_feature[i] = get_feature(use_cols)
    
    # return dict_feature

    if not os.path.exists('./feature/feature_dicts.dict'):
        feature_dicts = {}
        f = open('./feature/feature_dicts.dict','wb')
        pickle.dump(feature_dicts,f)
    f = open('./feature/feature_dicts.dict','rb')
    feature_dicts = pickle.load(f)
    feature_dicts[di] = dict_feature
    f = open('./feature/feature_dicts.dict','wb')
    pickle.dump(feature_dicts,f)

    # dump A
    if not os.path.exists('./feature/A_mat.dict'):
        dict_A = {}
        f = open('./feature/A_mat.dict','wb')
        pickle.dump(dict_A,f)
    f = open('./feature/A_mat.dict','rb')
    dict_A = pickle.load(f)
    dict_A[di] = A
    f = open('./feature/A_mat.dict','wb')
    pickle.dump(dict_A,f)

    # dump W
    if not os.path.exists('./feature/W_mat.dict'):
        dict_W = {}
        f = open('./feature/W_mat.dict','wb')
        pickle.dump(dict_W,f)
    f = open('./feature/W_mat.dict','rb')
    dict_W = pickle.load(f)
    dict_W[di] = W
    f = open('./feature/W_mat.dict','wb')
    pickle.dump(dict_W,f)