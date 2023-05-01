#!/usr/bin/env python3
import os.path
import random
import sys
import multiprocessing as mp
import numpy as np
import os
import random
import pandas as pd
sys.path.insert(1, '/home/wjin/projects/Protein_structure/scripts')
from foldseek_analysis.training import extract_pdb_features

d1='/home/wjin/data3/AlphaFold2/PDBs'
d2='/home/wjin/data3/AlphaFold2/ContactMaps/'
d3='/home/wjin/data3/AlphaFold2/AlphaFold_graph_k{}KNN/'
def contact_norm(x, d0=4):
    return 2.0/(1+(max(d0,x)/d0))

def unit_vec(v):
    return v / np.linalg.norm(v)


def calc_angles(coords, i, j):

    CA = coords[:, 0:3]

    u_1 = unit_vec(CA[i]     - CA[i - 1])
    u_2 = unit_vec(CA[i + 1] - CA[i])
    u_3 = unit_vec(CA[j]     - CA[j - 1])
    u_4 = unit_vec(CA[j + 1] - CA[j])
    u_5 = unit_vec(CA[j]     - CA[i])

    cos_phi_12 = u_1.dot(u_2)
    cos_phi_34 = u_3.dot(u_4)
    cos_phi_15 = u_1.dot(u_5)
    cos_phi_35 = u_3.dot(u_5)
    cos_phi_14 = u_1.dot(u_4)
    cos_phi_23 = u_2.dot(u_3)
    cos_phi_13 = u_1.dot(u_3)

    d = np.linalg.norm(CA[i] - CA[j])
    seq_dist = (j - i).clip(-4, 4)

    return np.array([cos_phi_12, cos_phi_34,
                     cos_phi_15, cos_phi_35,
                     cos_phi_14, cos_phi_23,
                     cos_phi_13, d,
                     seq_dist])

def calc_angles_forloop_wh(coords, edge_list, valid_mask):
    n_res = coords.shape[0]
    n_edges = len(edge_list)
    out = np.full((n_edges, 9), np.nan, dtype=np.float32)
    # CA = coords[:, 0:3]

    t=0
    for i, j in edge_list:
        if (i==0) or (i==n_res - 1) or (j==0) or (j==n_res - 1) or (i==j):
            continue
        if valid_mask[i - 1] and valid_mask[i] and valid_mask[i + 1] and valid_mask[j - 1] and valid_mask[j] and valid_mask[j + 1]:
            out[t] = calc_angles(coords, np.int64(i), np.int64(j))

        t+=1

    new_valid_mask = ~np.isnan(out).any(axis=1)

    return out, new_valid_mask

feature_cache = {}  # path: (features, mask)
def encoder_features_wh(graph_edgelist_df, pdb_path, virt_cb):
    """
    Calculate 3D descriptors for each residue of a PDB file.
    """
    edge_list=list(zip(graph_edgelist_df[0].apply(int),graph_edgelist_df[1].apply(int)))

    feat = feature_cache.get(pdb_path, None)
    if feat is not None:
        return feat

    coords, valid_mask = extract_pdb_features.get_coords_from_pdb(pdb_path, full_backbone=True)
    coords = extract_pdb_features.move_CB(coords, virt_cb=virt_cb)

    features, valid_mask2 = calc_angles_forloop_wh(coords, edge_list, valid_mask)

    seq_dist = (np.array(edge_list)[:,0] - np.array(edge_list)[:,1])[:, np.newaxis]
    log_dist = np.sign(seq_dist) * np.log(np.abs(seq_dist) + 1)

    vae_features = np.hstack([features, log_dist])
    feature_cache[pdb_path] = vae_features, valid_mask2

    return vae_features, valid_mask2

def run(file, d1, d3, knn_k):
    print(file)
    graph_file_path=os.path.join(d3.format(knn_k), file)
    graph_edgelist_df=pd.read_csv(graph_file_path,header=None)
    pdb_path=os.path.join(d1,file.replace('_k{}KNNedgelistWithWeight.csv'.format(knn_k), '.pdb'))
    edge_features, valid_mask2 = encoder_features_wh(graph_edgelist_df, pdb_path, virtual_center)

    df_final = pd.concat([graph_edgelist_df, pd.DataFrame(edge_features)], axis=1) 
    df_final.to_csv(os.path.join(d3.format(knn_k), file.replace('_k{}KNNedgelistWithWeight.csv'.format(knn_k),'_k{}KNNedgelistWithWeight_3Di.csv'.format(knn_k))), index=False, header=False)


a=ALPHA=270
b=BETA=0
c=D=2
knn_k=30
virtual_center = (a, b, c)
#virtual_center = None

files=list(filter(lambda x: x.endswith('_k{}KNNedgelistWithWeight.csv'.format(knn_k)), os.listdir(d3.format(knn_k))))
# for file in files:
#     run(file, d1, d3)

random.shuffle(files)
pool=mp.Pool(processes=None)
results=[pool.apply_async(run, args = (file,d1,d3,knn_k,)) for file in files]
aaa=[p.get() for p in results]

