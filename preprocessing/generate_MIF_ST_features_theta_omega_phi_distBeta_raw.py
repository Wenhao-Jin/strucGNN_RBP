#!/usr/bin/env python

from sequence_models.pretrained import load_model_and_alphabet
model_file='/home/wjin/projects/Protein_structure/data/MIF_ST/mifst.pt'
model, collater = load_model_and_alphabet(model_file)

AF_dir='/home/wjin/data3/AlphaFold2/PDBs'
out_dir='/home/wjin/data2/protein_structures/MIF-ST_input'

import os
import random
import numpy as np
from sequence_models.pdb_utils import parse_PDB, process_coords
import torch
files=list(filter(lambda x:x.endswith('model_v2.pdb'), os.listdir(AF_dir)))
random.shuffle(files)
for file in files:
    prot='-'.join(file.split('-')[1:3])
    if os.path.exists(os.path.join(out_dir, file.replace('.pdb','_betaDist_omega_theta_phi.npz'))):
        continue
        
    coords, wt, _ = parse_PDB(os.path.join(AF_dir,'AF-{}-model_v2.pdb'.format(prot)))
    coords = {
            'N': coords[:, 0],
            'CA': coords[:, 1],
            'C': coords[:, 2]
        }
    dist, omega, theta, phi = process_coords(coords)
    np.savez(os.path.join(out_dir,file.replace('.pdb','_betaDist_omega_theta_phi.npz')), prot_name= prot, seq=wt, Cbeta_dist=dist,omega=omega, theta=theta, phi=phi)
