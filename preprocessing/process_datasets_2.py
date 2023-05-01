#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as T
import torchvision.transforms as transforms

class Protein_Structures_processed(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, root_dir='/home/wjin/data3/AlphaFold2/Torch_dataset/', contact_map_dir='/home/wjin/data3/AlphaFold2/ContactMaps/',seq_dir='/home/wjin/data2/proteins/uniport_data/canonical_seq/', transform = None):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.AAs=['V', 'A', 'M', 'N', 'L', 'W', 'Q', 'I', 'R', 'F', 'E', 'K', 'D', 'T', 'H', 'S', 'Y', 'C', 'P', 'G','U','O']
        ident=np.identity(len(self.AAs))
        self.aa_vec={self.AAs[i]: ident[i] for i in range(len(self.AAs))}
        self.contact_map_dir=contact_map_dir
        self.seq_dir=seq_dir
        self.root_dir='/home/wjin/data3/AlphaFold2/Torch_dataset/'
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def process(self,ID,file):
#         file2=os.path.join(self.contact_map_dir,'AF-'+ID+'-F1-model_v2.pdbContactMap_norm.csv')
#         file3=os.path.join(self.seq_dir,ID+'.fasta')
#         mat1=np.loadtxt(file2,delimiter=',')
#         mat1=mat1.reshape(mat1.shape[0],mat1.shape[1],1)
        
#         with open(file3) as f:
#             sequence=f.read().split('\n')[1]
            
#         if len(sequence)!=mat1.shape[0]:
#             raise ValueError('The sequence and ContactMap matrix (normed) of {} have unmatched protein length.'.format(ID))
        
#         tmp_mat=np.zeros([len(sequence),len(sequence),2*len(self.AAs)])
#         for i in range(len(sequence)):
#             for j in range(i,len(sequence)):
#                 if i==j:
#                     tmp_mat[i,j]=np.concatenate([self.aa_vec[sequence[i]],self.aa_vec[sequence[j]]])
#                 else:
#                     tmp_mat[i,j]=np.concatenate([self.aa_vec[sequence[i]],self.aa_vec[sequence[j]]])
#                     tmp_mat[j,i]=np.concatenate([self.aa_vec[sequence[j]],self.aa_vec[sequence[i]]])
                    
#         #print(mat1.shape, tmp_mat.shape)
#         mat=np.concatenate([mat1,tmp_mat],axis=2)
        file0=os.path.join(self.root_dir,'AF-'+ID+'-F1-model_v2.pdbContactMap_norm.dt')
        mat=torch.load(file0)
        if self.transform:
            mat,y = self.transform((mat,0))
        
        torch.save(mat, file)
                    
        
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        
        # Load data and get label
        file=os.path.join(self.root_dir,'AF-'+ID+'-F1-model_v2.pdbContactMap_norm_Cropped.dt')
        if not os.path.exists(file):
            try:
                self.process(ID,file)
            except ValueError as err:
                warnings.warn(str(err))
                return 'No data','No data'
            
        X = torch.load(file)
        y = self.labels[index]
#         X = torch.from_numpy(X) if type(X)!=torch.Tensor else X
#         y = torch.from_numpy(np.array([self.labels[index]])).int()
        
        
#         if self.transform:
#             X,y = self.transform((X,y))

        return X, y


    
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
            self.output_prot_len = output_size
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            self.output_prot_len = output_size[0]

    def __call__(self, sample):
        X, y = sample
        prot_len=X.shape[0]
        if prot_len > self.output_prot_len:
            h, w = X.shape[:2]
            new_h, new_w = self.output_size
            #print(h,new_h,w,new_w)
            #print('X', type(X))
            top = torch.randint(0, h - new_h, (1,))[0]
            left = torch.randint(0, w - new_w, (1,))[0]
            #print('random int:',top,left)
            #top = np.random.randint(0, h - new_h)
            #left = np.random.randint(0, w - new_w)
            X_new = X[top: top + new_h,
                          left: left + new_w]
        else:
            X_new = X
            #X_new = torch.from_numpy(np.array(X))
            
        #print('X_new',type(X_new))
        return X_new, y

class Padding(object):
    """Padding the 2D matrix with [0,0,0..] values to a certain size.
    
    Args:
        output_prot_len (int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_prot_len):
        assert isinstance(output_prot_len, (int, float))
        if isinstance(output_prot_len, int):
            self.output_prot_len = output_prot_len
        else:
            self.output_prot_len = int(round(output_prot_len))

    def __call__(self, sample, padding=0):
        X, y = sample
        prot_len=X.shape[0]
        if prot_len<self.output_prot_len:
            X_padded=np.pad(X, pad_width=((0,self.output_prot_len-prot_len),(0,self.output_prot_len-prot_len),(0,0)))
            X_padded=torch.from_numpy(np.array(X_padded))
        else:
            X_padded=X
            
        #print('X_padded',type(X_padded))
        return X_padded, y
                
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        X, y = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        #X = X.transpose((2, 0, 1))
        X = np.transpose(X,(2, 0, 1))
        #print(type(X))
        #print('X_final',type(X))
        return X, torch.from_numpy(np.array([y]))
        #return torch.from_numpy(X), torch.from_numpy(np.array([y]))


IDs=list(filter(lambda x: x.endswith('.pdbContactMap_norm.dt'), os.listdir('/home/wjin/data3/AlphaFold2/Torch_dataset/')))
IDs=list(map(lambda x: x.split('-')[1], IDs))
import random
random.shuffle(IDs)
transformed_dataset=Protein_Structures_processed(IDs,[0]*len(IDs),transform=transforms.Compose([
                                               RandomCrop(512),
                                               Padding(512),
                                               ToTensor()
                                           ]))
dataloader = DataLoader(transformed_dataset, batch_size=1,
                        shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch)
    a=1
