import pandas as pd
import numpy as np
import os
import math
import torch
from torch import Tensor
from torch import nn
import logging
import pickle
import collections
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from ..utils.train_val import train_MIF, val_MIF

root='/home/wjin/data2/protein_structures/MIF-ST_input/'
HydRa_df=pd.read_csv('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/SONAR_plus_output/SONAR+/HydRa2.0_proteins_with_RBP_categories_RBPflagCorrected_Nov2021_new_addOccDomainProtLenNormed_sigOccThres0.05_0.001.csv')
IDs=list(map(lambda x: '-'.join(x.split('-')[1:3]), filter(lambda x: x.endswith('_MIF-ST_input.npz'), os.listdir(root))))

HydRa_df=HydRa_df.drop_duplicates('Unnamed: 0')
HydRa_df=HydRa_df.set_index('Unnamed: 0')
RBP_set=set(HydRa_df[HydRa_df.RBP_flag==1].index)
# f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/TrainEvaluation_protein_uniprotIDs_menthaBioPlexSTRING.txt')
# train_prots=set(f.read().split('\n'))
# f.close()
# train_prot_list=train_prots
# train_prot_list=list(set(train_prot_list).intersection(HydRa_df.index))
# train_ind=list(set(train_prot_list))
# train_IDs=list(filter(lambda x:x.split('-')[0] in train_ind, IDs))
# train_labels=[1 if i.split('-')[0] in RBP_set else 0 for i in train_IDs]

f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/TrainEvaluation_protein_uniprotIDs_menthaBioPlexSTRING_TrainSet.txt')
train_prots=set(f.read().split('\n'))
f.close()
train_prot_list=train_prots
train_prot_list=list(set(train_prot_list).intersection(HydRa_df.index))
train_ind=list(set(train_prot_list))
train_ids=list(filter(lambda x:x.split('-')[0] in train_ind, IDs))
train_labels=[1 if i.split('-')[0] in RBP_set else 0 for i in train_ids]
train_labs=train_labels

f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/TrainEvaluation_protein_uniprotIDs_menthaBioPlexSTRING_ValSet.txt')
val_prots=set(f.read().split('\n'))
f.close()
val_prot_list=val_prots
val_prot_list=list(set(val_prot_list).intersection(HydRa_df.index))
val_ind=list(set(val_prot_list))
val_ids=list(filter(lambda x:x.split('-')[0] in val_ind, IDs))
val_labels=[1 if i.split('-')[0] in RBP_set else 0 for i in val_ids]
val_labs=val_labels

pos_count=len([x for x in train_labels if x==1])+len([x for x in val_labels if x==1])
pos_weight=(len(train_labels)+len(val_labels)-pos_count)*1.0/pos_count

## Prepare test dataset.
f=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/Test_protein_uniprotIDs_menthaBioPlexSTRING.txt')
test_prots=set(f.read().split('\n'))
f.close()
test_ind=test_prots
#test_ind=set(filter(lambda x: os.path.exists('/home/wjin/data2/proteins/uniport_data/canonical_seq/'+x+'.spd3'),test_prots))
f1=open('/home/wjin/projects/RBP_pred/RBP_identification/Data_new/data/Test_proteins_to_be_removed.txt')
test_to_be_removed=set(f1.read().split('\n'))
f1.close()
test_ind=list(test_ind-test_to_be_removed) # Remove the test samples that are highly similar (by sequence) to those in training set. (cd-hit, >90% identity, >90% length)
test_ind=list(set(test_ind).intersection(HydRa_df.index))
test_IDs=list(filter(lambda x:x.split('-')[0] in test_ind, IDs))
test_labels=[1 if i.split('-')[0] in RBP_set else 0 for i in test_IDs]


##### Create DataLoader for training, validation and testing.
batch_size=1

transformed_dataset_training = MIFDataset(train_ids,train_labels)
transformed_dataset_val = MIFDataset(val_ids,val_labels)

trainloader = DataLoader(transformed_dataset_training, batch_size=batch_size, shuffle=True)

valloader = DataLoader(transformed_dataset_val, batch_size=batch_size, shuffle=True)

transformed_dataset_test = MIFDataset(test_IDs, test_labels)

testloader = DataLoader(transformed_dataset_test, batch_size=batch_size, shuffle=True)


##### Fine-tune Step1

## Freeze MIF and train added FC layers, Finetune step1, freeze other layers except the added FC layers
## Global_mean_pool
## With early stopping

import torch.optim as optim


hidden_fc_feats=[512,256,16]
output_channel=1
has_fc_bias=True
globalpool=global_mean_pool
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#net=MIF_ST_RBP_2(model, hidden_fc_feats, output_channel, has_fc_bias=has_fc_bias, globalpool=globalpool)
pause_path='/home/wjin/projects/Protein_structure/data/models/MIF_ST_RBP_BCEcorrected_FinetuneStep1_best_checkpoint.pth'
net = torch.load(pause_path)   
net=net.to(device)
net=net.double()
#net=net.float()

#criterion = nn.CrossEntropyLoss()
pos_weight=torch.from_numpy(np.array([pos_weight]))
pos_weight=pos_weight.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

min_val_loss=np.Inf
early_stopping_n_epoches=6
epochs_no_improve=0
train_loss_history=[]
val_loss_history=[]
#PATH_b='/home/wjin/projects/Protein_structure/data/models/GNN_ProteinBERT-RBP_PyG_RBP_model_best_checkpoint.pth'
PATH_entire_b='/home/wjin/projects/Protein_structure/data/models/MIF_ST_RBP_BCEcorrected_FinetuneStep1_best_checkpoint.pth' ## The MIF-ST was frozen, only the added FC layers are trained.

for epoch in range(90):  # loop over the dataset multiple times
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
    net, train_loss=train_MIF(net, trainloader,optimizer, criterion)
    train_loss, roc_auc_train=val_MIF(net, trainloader, criterion)
    val_loss, roc_auc=val_MIF(net, valloader, criterion)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    print('Epoch: {:3d}, Train_loss: {:.4f}, ROC-AUC on train: {}, Val_loss: {:.4f}, ROC-AUC on val: {}'.format(epoch, train_loss, roc_auc_train, val_loss, roc_auc))
    if val_loss < min_val_loss:
        min_val_loss=val_loss
        torch.save(net, PATH_entire_b)
        #torch.save(net.state_dict(), PATH_b)
        epochs_no_improve=0
        
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve>early_stopping_n_epoches:
        break
        
#     end.record()
#     # Waits for everything to finish running
#     torch.cuda.synchronize()
#     print(start.elapsed_time(end))
        
print('Finished Training')

PATH_entire = '/home/wjin/projects/Protein_structure/data/models/MIF_ST_RBP_BCEcorrected_FinetuneStep1.pth'
torch.save(net, PATH_entire)

# PATH = '/home/wjin/projects/Protein_structure/data/models/GNN_ProteinBERT-RBP_PyG_RBP_model.pth'
# torch.save(net.state_dict(), PATH)


##### Fine-tune Step2
## Unfreeze the layers in MIF-RBP and run fine-tune step2.
#PATH_entire='/home/wjin/projects/Protein_structure/data/models/MIF_ST_RBP_FinetuneStep2_rep2_valAUC0.8197_valloss0.9425.pth'
PATH_entire='/home/wjin/projects/Protein_structure/data/models/MIF_ST_RBP_BCEcorrected_FinetuneStep1.pth'
model = torch.load(PATH_entire)
for child in model.children():
    for param in child.parameters():
        param.requires_grad = True
        
for child in model.children():
    print(child)
    for param in child.parameters():
        print(param.requires_grad )        
        
## Finetune on all layers
## Global_mean_pool
## With early stopping

import torch.optim as optim


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net=model.to(device)
net=net.double()
#net=net.float()

#criterion = nn.CrossEntropyLoss()
pos_weight=torch.from_numpy(np.array([pos_weight]))
pos_weight=pos_weight.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=0.000005)
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#min_val_loss=0.9425
early_stopping_n_epoches=3
epochs_no_improve=0
train_loss_history=[]
val_loss_history=[]
#PATH_b='/home/wjin/projects/Protein_structure/data/models/GNN_ProteinBERT-RBP_PyG_RBP_model_best_checkpoint.pth'
PATH_entire_b='/home/wjin/projects/Protein_structure/data/models/MIF_ST_RBP_BCEcorrected_FinetuneStep2_best_checkpoint.pth' ## The MIF-ST was frozen, only the added FC layers are trained.

for epoch in range(10):  # loop over the dataset multiple times
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
    net, train_loss=train_MIF(net, trainloader,optimizer, criterion)
    train_loss, roc_auc_train=val_MIF(net, trainloader, criterion)
    val_loss, roc_auc=val_MIF(net, valloader, criterion)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    print('Epoch: {:3d}, Train_loss: {:.4f}, ROC-AUC on train: {}, Val_loss: {:.4f}, ROC-AUC on val: {}'.format(epoch, train_loss, roc_auc_train, val_loss, roc_auc))
    if val_loss < min_val_loss:
        min_val_loss=val_loss
        torch.save(net, PATH_entire_b)
        #torch.save(net.state_dict(), PATH_b)
        epochs_no_improve=0
        
    else:
        epochs_no_improve += 1
        
    if epochs_no_improve>early_stopping_n_epoches:
        break
        
#     end.record()
#     # Waits for everything to finish running
#     torch.cuda.synchronize()
#     print(start.elapsed_time(end))
        
print('Finished Training')

PATH_entire = '/home/wjin/projects/Protein_structure/data/models/MIF_ST_RBP_BCEcorrected_FinetuneStep2.pth'
torch.save(net, PATH_entire)

# PATH = '/home/wjin/projects/Protein_structure/data/models/GNN_ProteinBERT-RBP_PyG_RBP_model.pth'
# torch.save(net.state_dict(), PATH)
