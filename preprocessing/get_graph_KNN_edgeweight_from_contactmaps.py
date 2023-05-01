#!/usr/bin/env python
import multiprocessing as mp
import numpy as np
import os
import random
import pandas as pd
#d2='/home/wjin/data3/AlphaFold2/ContactMaps/'
d2='/home/wjin/data3/AlphaFold2/ContactMaps_DeepFRI/'
d3='/home/wjin/data3/AlphaFold2/AlphaFold_graph_k{}KNN/'
def contact_norm(x, d0=4):
    return 2.0/(1+(max(d0,x)/d0))

def get_knn_edgeWithWeight_list(contactMap, k=10):
    #m=np.array(contactMap_df)
    m=contactMap
    a=np.argsort(m)[:,:k]
    pairs=[(i,e) for i in range(len(a)) for e in a[i]] 
    b=[[contact_norm(m[x,y])] for x,y in pairs]
    return pd.DataFrame(np.concatenate([pairs,b],axis=1))

def get_edgelist_from_ContactMap(f,d2,d3,k=10):
    if not(os.path.exists(os.path.join(d2,f.replace('.ContactMap.npz','_edgelist.csv')))):
        #contactMap_df=pd.read_csv(os.path.join(d2,f),header=None)
        d=np.load(os.path.join(d2,f))
        df=get_knn_edgeWithWeight_list(d['C_alpha'], k=k)
        #df.columns=['node1','node2','Contact_weight']
        df.to_csv(os.path.join(d3.format(k),f.replace('.ContactMap.npz','_k{}KNNedgelistWithWeight.csv'.format(k))),index=False,header=False)

files=list(filter(lambda x: x.endswith('ContactMap.npz'),os.listdir(d2)))
random.shuffle(files)
pool=mp.Pool(processes=None)
k=15
results=[pool.apply_async(get_edgelist_from_ContactMap, args = (f,d2,d3,k,)) for f in files]
aaa=[p.get() for p in results]

