# multiprocess
import random
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, truncexpon, genpareto
from typing import Dict, Any
from numpy.random import choice
from tqdm import tqdm
from physical_db import DBConnection, TrueCardinalityEstimator
import time
import pickle
from syndata_utils import Pnode, Fnode, Add_cols, Get_schema, table2pg, Sql_truecard, get_join, get_features, get_usecols, sql_trans
from multiprocessing import Process
import os


def run_di(num1,num2):
    for di in range(num1,num2):
        #  Initialization at the beginning of each data set
        num_pnode = random.randint(1,2)  # Number of primary nodes
        if num_pnode == 1:
            num_fnode = random.randint(0,3)  # Number of secondary nodes
        else:
            num_fnode = random.randint(2,3)  #  Number of secondary nodes

        # test
        # num_pnode = 1
        # num_fnode = 0
        
        sun_node = num_pnode + num_fnode
        A = np.zeros((sun_node, sun_node))  # adjacent matrix
        W = np.zeros((sun_node, sun_node))  # Edge weight matrix
        dict_node = {}  
        # dict_datatime = {}
        # main
        # di = 0 
        Pnode(num_pnode,dict_node,A)
        Fnode(num_fnode,num_pnode,dict_node,A,W)
        Add_cols(num_fnode,num_pnode,dict_node)
        schema_sql = Get_schema(num_fnode,num_pnode,di,dict_node)
        table2pg(schema_sql, num_fnode, num_pnode, di, dict_node)

        # Generate and run SQL
        sql_num = 15000   # modify
        Sql_truecard(di, A, dict_node, sql_num, num_pnode)

        get_join(A,di,num_pnode,dict_node)

        # Output: dict
        get_features(dict_node,A,W,di)


        sql_trans(A,di,num_pnode,dict_node)


if __name__=='__main__':
    # print('Parent process %s.' % os.getpid())
    
    '''p1 = Process(target=run_di, args=(0,50,))
    p2 = Process(target=run_di, args=(50,100,))
    p3 = Process(target=run_di, args=(100,150,))
    p4 = Process(target=run_di, args=(150,200,))
    p5 = Process(target=run_di, args=(200,250,))
    p6 = Process(target=run_di, args=(250,300,))
    p7 = Process(target=run_di, args=(300,350,))
    p8 = Process(target=run_di, args=(350,400,))

    p9 = Process(target=run_di, args=(400,450,))
    p10 = Process(target=run_di, args=(450,500,))
    p11 = Process(target=run_di, args=(500,550,))
    p12 = Process(target=run_di, args=(550,600,))
    p13 = Process(target=run_di, args=(600,650,))
    p14 = Process(target=run_di, args=(650,700,))
    p15 = Process(target=run_di, args=(700,750,))
    p16 = Process(target=run_di, args=(750,800,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    p9.start()
    p10.start()
    p11.start()
    p12.start()
    p13.start()
    p14.start()
    p15.start()
    p16.start()
    '''

    p1 = Process(target=run_di, args=(800,850,))
    p2 = Process(target=run_di, args=(850,900,))
    p3 = Process(target=run_di, args=(900,950,))
    p4 = Process(target=run_di, args=(950,1000,))
    p5 = Process(target=run_di, args=(1000,1050,))
    p6 = Process(target=run_di, args=(1050,1100,))
    p7 = Process(target=run_di, args=(1100,1150,))
    p8 = Process(target=run_di, args=(1150,1200,))

    p1.start()
    #p2.start()
    #p3.start()
    #p4.start()
    #p5.start()
    #p6.start()
    #p7.start()
    #p8.start()
    