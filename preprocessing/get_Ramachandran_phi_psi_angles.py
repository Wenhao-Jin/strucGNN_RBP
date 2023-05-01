#!/usr/bin/env python
import numpy as np
import Bio.PDB
import os
import random
AF_dir='/home/wjin/data3/AlphaFold2/PDBs'
out_dir='/home/wjin/data2/protein_structures/Ramachandran_phi_psi_angles'
files=list(filter(lambda x:x.endswith('model_v2.pdb'), os.listdir(AF_dir)))
random.shuffle(files)
for file in files:
    prot='-'.join(file.split('-')[1:3])
    if not os.path.exists(os.path.join(out_dir, file.replace('.pdb','_phi_psi.npz'))):
        pdb=Bio.PDB.PDBParser().get_structure(prot, os.path.join(AF_dir,file))
        polypeptides = Bio.PDB.PPBuilder().build_peptides(pdb)
        #assert len(polypeptides)==1
        phi_psi_list=[]
        for poly_index, poly in enumerate(polypeptides):
            phi_psi_list+=poly.get_phi_psi_list()
        
        phi_psi = np.array(phi_psi_list)
        np.savez(os.path.join(out_dir,file.replace('.pdb','_phi_psi.npz')), phi=phi_psi[:,0], psi=phi_psi[:,1])
