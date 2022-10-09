import numpy as np
import pandas as pd 
import torch.nn.functional as F
import pickle
import torch
import time
import math

f_feature = open('../data/datasets/Synthetic/feature/feature_dicts.dict','rb')
feat_dict = pickle.load(f_feature)

# len_feat = 47
len_feat = len(feat_dict[0][0])
tar = [0]*len_feat*5
for id, v in enumerate(feat_dict[1].values()):
    tar[id*len_feat:(id+1)*len_feat] = v

train_vecs = list()
test_vecs = list()


for i in range(0,1000):
    tar = [0]*len_feat*5
    for id, v in enumerate(feat_dict[i].values()):
        tar[id*len_feat:(id+1)*len_feat] = v
    train_vecs.append(tar)


for i in range(1000,1260):
    tar = [0]*len_feat*5
    for id, v in enumerate(feat_dict[i].values()):
        tar[id*len_feat:(id+1)*len_feat] = v
    test_vecs.append(tar)

train_vecs = torch.tensor(train_vecs).to('cuda')
test_vecs = torch.tensor(test_vecs).to('cuda')

# print(train_vecs, test_vecs)

# for i in range(len(test_vecs)):
res = []
for i in range(len(test_vecs)):
    eud = F.pairwise_distance(test_vecs[i],train_vecs,p=2)
    res.append(int(eud.argmin()))

f_res = open('../exp/res/res_260_knn.list', 'wb')
pickle.dump(res, f_res)
print(res)
f_res.close()
