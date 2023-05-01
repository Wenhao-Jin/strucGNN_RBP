#!/usr/bin/env python
import multiprocessing as mp
import numpy as np
import os
import random
import pandas as pd
npz_dir='/home/wjin/data3/AlphaFold2/ContactMaps_DeepFRI/'

def contact_norm(x, d0=4):
    return 2.0/(1+(max(d0,x)/d0))

def get_normalized_ContactMap(f,npz_dir,out_dir):
    out_fn=os.path.join(out_dir,f.replace('ContactMap.npz','ContactMap_addNorm.npz'))
    if not(os.path.exists(out_fn)):
        cmap = np.load(os.path.join(npz_dir, f))
        df=pd.DataFrame(cmap['C_alpha'])
        df=df.applymap(lambda x: contact_norm(x))
        np.savez_compressed(out_fn,
                            C_alpha=cmap['C_alpha'],
                            C_alpha_norm=np.array(df),
                            seqres=cmap['seqres'],
                            )

out_dir=npz_dir
files=list(filter(lambda x: x.endswith('ContactMap.npz'),os.listdir(npz_dir)))
random.shuffle(files)
pool=mp.Pool(processes=4)
results=[pool.apply_async(get_normalized_ContactMap, args = (f,npz_dir,out_dir,)) for f in files]
aaa=[p.get() for p in results]

