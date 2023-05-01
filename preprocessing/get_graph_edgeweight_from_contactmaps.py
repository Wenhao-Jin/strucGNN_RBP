#!/usr/bin/env python
import multiprocessing as mp
import numpy as np
import os
import random
import pandas as pd
d2='/home/wjin/data3/AlphaFold2/ContactMaps/'
d3='/home/wjin/data3/AlphaFold2/AlphaFold_graph_10A/'
def contact_norm(x, d0=4):
    return 2.0/(1+(max(d0,x)/d0))

def get_edgeWithWeight_list(contactMap_df, thres=10):
    m=np.array(contactMap_df)
    a=np.argwhere(m<thres)
    b=[[contact_norm(m[x,y])] for x,y in a]
    return pd.DataFrame(np.concatenate([a,b],axis=1))

def get_edgelist_from_ContactMap(f,d2,d3):
    if not(os.path.exists(os.path.join(d2,f.replace('ContactMap.csv','_edgelist.csv')))):
        contactMap_df=pd.read_csv(os.path.join(d2,f),header=None)
        df=get_edgeWithWeight_list(contactMap_df, thres=10)
        #df.columns=['node1','node2','Contact_weight']
        df.to_csv(os.path.join(d3,f.replace('ContactMap.csv','_edgelistWithWeight.csv')),index=False,header=False)

files=list(filter(lambda x: x.endswith('ContactMap.csv'),os.listdir(d2)))
random.shuffle(files)
pool=mp.Pool(processes=None)
results=[pool.apply_async(get_edgelist_from_ContactMap, args = (f,d2,d3,)) for f in files]
aaa=[p.get() for p in results]

