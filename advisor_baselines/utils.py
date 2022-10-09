
# utils.py
import pickle
import pandas as pd
from tqdm import tqdm
import torch
import math

def process_metric(lable_acc_dict,lable_eff_dict,df):
    df = df.set_index(['Index'])
    for i in range(len(df)):
        acc = df.loc[f'table{i}']['Qerror_mean']  # + df.loc[f'table{i}']['Qerror_mean']
        if acc>15:
            acc = 15
        eff = math.log(df.loc[f'table{i}']['Evalu_time']+5)  # 
        lable_acc_dict[i].append(acc)
        lable_eff_dict[i].append(eff)  # 
    return lable_acc_dict,lable_eff_dict
    
def Norm(lable_acc_dict,lable_eff_dict):
    for i in range(len(lable_acc_dict)):
        tmp = np.array(lable_acc_dict[i])
        
        lable_acc_dict[i] = list((max(tmp)-tmp)/(max(tmp)-min(tmp)))
        # print(i,lable_acc_dict[i])
        tmp = np.array(lable_eff_dict[i])
        lable_eff_dict[i] = list((max(tmp)-tmp)*0.75/(max(tmp)-min(tmp)))
    return lable_acc_dict,lable_eff_dict


def preprocess():
    f_A = open('../data/feature/A_mat.dict','rb')
    A_dict = pickle.load(f_A)

    f_W = open('../data/feature/W_mat.dict','rb')
    W_dict = pickle.load(f_W)

    f_feature = open('../data/feature/feature_dicts.dict','rb')
    feat_dict = pickle.load(f_feature)

    lable_acc_dict = {}
    lable_eff_dict = {}
    for i in range(len(A_dict)):
        lable_acc_dict[i] = []
        lable_eff_dict[i] = []

    df_bayescard = pd.read_csv('../models/Metric_res/bayescard.csv')
    df_deepdb = pd.read_csv('../models/Metric_res/deepdb.csv')
    df_mscn = pd.read_csv('../models/Metric_res/mscn.csv')
    df_naru = pd.read_csv('../models/Metric_res/naru.csv')
    df_uae = pd.read_csv('../models/Metric_res/uae.csv')
    df_nn = pd.read_csv('../models/Metric_res/nn.csv')
    df_xgb = pd.read_csv('../models/Metric_res/xgb.csv')

    # max_acc,min_acc,max_eff,min_eff = 0,1e4,0,1e4

    # print('max_acc, min_acc, max_eff, min_eff:',max_acc, min_acc, max_eff, min_eff)
    for df in [df_bayescard,df_deepdb,df_mscn,df_naru,df_uae,df_nn,df_xgb]:
        lable_acc_dict,lable_eff_dict = process_metric(lable_acc_dict,lable_eff_dict,df)
    lable_acc_dict,lable_eff_dict = Norm(lable_acc_dict,lable_eff_dict) 

    return A_dict,W_dict,feat_dict,lable_acc_dict,lable_eff_dict

# A_dict,W_dict,feat_dict,lable_acc_dict,lable_eff_dict = preprocess()


# lables  [0,1,2,3,4,5] [bayescard,deepdb,mscn,naru,uae,nn,xgb]
def lable2cate(lable):
    return lable.index(max(lable))





import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

def sig(x):
    if x>0.9:
        return 0
    elif x<0.7:
        return 1
    else:
        return 0.5


def s_max_min(list_all_label):
    smax = 0
    smin = 1
    for label1 in tqdm(list_all_label):
        for label2 in list_all_label:
            # print(label1,label2)
            scos = cosine_similarity( label1.reshape(1, -1), label2.reshape(1, -1) )
            # eucd = euclidean_distances( label1.reshape(1, -1), label2.reshape(1, -1) )
            sim = scos # + 1 - eucd
            # print(sim)
            if np.max(sim)>smax:
                smax = np.max(sim)
            if np.min(sim)<smin:
                smin = np.min(sim)
    return smax, smin


# o1,o2:mat  return:mat
def y2Y(y1,y2,smax,smin):
    Y = []
    y1 = torch.Tensor.cpu(y1)
    y2 = torch.Tensor.cpu(y2)
    for i in range(len(y1)):
        y1_ = y1[i].reshape(1,-1)
        y2_ = y2[i].reshape(1,-1)
        scos = cosine_similarity( y1_,y2_ )
        # eucd = euclidean_distances( y1_,y2_ )
        sim = scos[0][0] # + 1 - eucd[0][0]
        sim = (sim-smin)/(smax-smin)
        # sim -= 0.18
        Y_ = sig(sim)

        Y.append(Y_)
    # Y = torch.tensor.gpu(Y)
    Y = np.array(Y)  # (batch_size*1)
    return Y
