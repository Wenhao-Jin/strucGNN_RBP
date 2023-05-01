import pandas as pd
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ..utils.train_val import train, val
from ..strucGNNs.models_strucGNN1 import *

root='/home/wjin/data/RBP_pred/protein_structure/PyG_dataset/'

HydRa_df=pd.read_csv('/home/wjin/data/RBP_pred/HydRa/data/score_tables/HydRa2.0_proteins_with_RBP_categories_RBPflagCorrected_Nov2021_new_addOccDomainProtLenNormed_sigOccThres0.05_0.001.csv')
IDs=list(map(lambda x: '-'.join(x.split('-')[1:3]), filter(lambda x: x.endswith('_coords.csv'), os.listdir(root+'raw'))))

HydRa_df=HydRa_df.drop_duplicates('Unnamed: 0')
HydRa_df=HydRa_df.set_index('Unnamed: 0')
RBP_set=set(HydRa_df[HydRa_df.RBP_flag==1].index)

f=open('/home/wjin/data/RBP_pred/HydRa/data/files/TrainEvaluation_protein_uniprotIDs_menthaBioPlexSTRING_TrainSet.txt')
train_prots=set(f.read().split('\n'))
f.close()
train_prot_list=train_prots
train_prot_list=list(set(train_prot_list).intersection(HydRa_df.index))
train_ind=list(set(train_prot_list))
train_ids=list(filter(lambda x:x.split('-')[0] in train_ind, IDs))
train_labels=[1 if i.split('-')[0] in RBP_set else 0 for i in train_ids]
train_labs=train_labels

f=open('/home/wjin/data/RBP_pred/HydRa/data/files/TrainEvaluation_protein_uniprotIDs_menthaBioPlexSTRING_ValSet.txt')
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
f=open('/home/wjin/data/RBP_pred/HydRa/data/files/Test_protein_uniprotIDs_menthaBioPlexSTRING.txt')
test_prots=set(f.read().split('\n'))
f.close()
test_ind=test_prots
#test_ind=set(filter(lambda x: os.path.exists('/home/wjin/data2/proteins/uniport_data/canonical_seq/'+x+'.spd3'),test_prots))
f1=open('/home/wjin/data/RBP_pred/HydRa/data/files/Test_proteins_to_be_removed.txt')
test_to_be_removed=set(f1.read().split('\n'))
f1.close()
test_ind=list(test_ind-test_to_be_removed) # Remove the test samples that are highly similar (by sequence) to those in training set. (cd-hit, >90% identity, >90% length)
test_ind=list(set(test_ind).intersection(HydRa_df.index))
test_IDs=list(filter(lambda x:x.split('-')[0] in test_ind, IDs))
test_labels=[1 if i.split('-')[0] in RBP_set else 0 for i in test_IDs]


##### Set up DataLoader for training, validation and testing.

batch_size = 8

embed_dir='/home/wjin/data/RBP_pred/protein_structure/AA_Embedding/ProteinBERT-RBP_emb_trainingset/'
embed_name='ProteinBERT-RBP-trainingSet'

transformed_dataset_training = Protein_Structures_PyG(root,train_ids,train_labs, embed_dir=embed_dir,embed_name=embed_name)


transformed_dataset_val = Protein_Structures_PyG(root,val_ids,val_labs, embed_dir=embed_dir,embed_name=embed_name)

trainloader = DataLoader(transformed_dataset_training, batch_size=batch_size,
                         shuffle=True, num_workers=0)

valloader = DataLoader(transformed_dataset_val, batch_size=batch_size,
                         shuffle=True, num_workers=0)

transformed_dataset_test = Protein_Structures_PyG(root, test_IDs, test_labels, embed_dir=embed_dir,embed_name=embed_name)

testloader = DataLoader(transformed_dataset_test, batch_size=batch_size,
                         shuffle=True, num_workers=0)




##### Train the model

import sys
import logging

knn_k=15

so = open("/home/wjin/data/RBP_pred/protein_structure/models_PyG2.1.0/PointConv_ProteinBERT-RBP-trainingSet_training_knn_k{}.log".format(knn_k), 'w', 10)
sys.stdout.echo = so
sys.stderr.echo = so

get_ipython().log.handlers[0].stream = so
get_ipython().log.setLevel(logging.INFO)

import torch.optim as optim

gnn_layer='PointConv'


is_EGNN=False
in_node_channel=128
hidden_fc_feats=[1024]
#edge_embedding_dims=[16,16,16]
edge_attr_dim=1
pos_dim=3
has_gnn_bias=False
knn_k=15

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = PointTransformerNet(in_node_channel, out_channels=1,
                dim_model=[32, 64, 128], k=knn_k).to(device)
net=torch.load('/home/wjin/data/RBP_pred/protein_structure/models_PyG2.1.0/PointTransformer_k15_ProteinBERT-RBP-trainingset_corrected_PyG_RBP_EntireModel_best_checkpoint_ValAUC0.841_ValLoss0.775.pth')

net=net.to(device)
net=net.double()

#criterion = nn.CrossEntropyLoss()
pos_weight=torch.from_numpy(np.array([pos_weight]))
pos_weight=pos_weight.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#min_val_loss=np.Inf
min_val_loss=0.7754
early_stopping_n_epoches=5
epochs_no_improve=0
train_loss_history=[]
val_loss_history=[]
PATH_entire_b='/home/wjin/data/RBP_pred/protein_structure/models_PyG2.1.0/PointTransformer_k{}_ProteinBERT-RBP-trainingset_corrected_PyG_RBP_EntireModel_best_checkpoint.pth'.format(knn_k)  

for epoch in range(20):  # loop over the dataset multiple times
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     train_loss=train(trainloader,optimizer,EGNN=is_EGNN)
#     val_loss, roc_auc=val(valloader,EGNN=is_EGNN)
    net, train_loss=train(net, trainloader,optimizer, criterion, EGNN=is_EGNN)
    val_loss, roc_auc=val(net, valloader, criterion, EGNN=is_EGNN)
    
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    #print('Epoch: {:3d}, Train_loss: {:.4f}, Val_loss: {:.4f}, ROC-AUC on val: {}'.format(epoch, train_loss, val_loss, roc_auc))
    print('Epoch: {:3d}, Train_loss: {:.4f}, Val_loss: {:.4f}, ROC-AUC on val: {}'.format(epoch, train_loss, val_loss, roc_auc))
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

PATH_entire = '/home/wjin/data/RBP_pred/protein_structure/models_PyG2.1.0/PointTransformer_k{}_ProteinBERT-RBP-trainingset_corrected_PyG_RBP_EntireModel.pth'.format(knn_k)  
torch.save(net, PATH_entire)

# PATH = '/home/wjin/projects/Protein_structure/data/models/GNN_ProteinBERT-RBP-trainingset_PyG_RBP_model.pth'
# torch.save(net.state_dict(), PATH)




## Model evaluation with Test Set
gnn_layer='PointTransformer'
knn_k=15
is_EGNN=False
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pos_weight=torch.from_numpy(np.array([pos_weight]))
pos_weight=pos_weight.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

PATH_entire='/home/wjin/data/RBP_pred/protein_structure/models_PyG2.1.0/PointTransformer_k15_ProteinBERT-RBP-trainingset_corrected_PyG_RBP_EntireModel_best_checkpoint.pth'
out_dir='/home/wjin/data/RBP_pred/protein_structure/predictions_PyG2.1.0'
model_name_prefix=embed_name+'_'+gnn_layer+'_k'+str(knn_k)
net = torch.load(PATH_entire)
## On validation set
val_loss, val_roc_auc, prot_names, y_val_pred, y_val_true =val(net, valloader, criterion, OnTestSet=True, EGNN=is_EGNN)
pickle.dump({'proteins': np.asarray(prot_names),
                     'Y_pred': y_val_pred,
                     'Y_true': y_val_true},
                    open(os.path.join(out_dir,model_name_prefix + '_corrected_results_onValSet_best_checkpoint.pckl'), 'wb'))

## On test set
test_loss, test_roc_auc, prot_names, y_test_pred, y_test_true =val(net, testloader, criterion, OnTestSet=True, EGNN=is_EGNN)
pickle.dump({'proteins': np.asarray(prot_names),
                     'Y_pred': y_test_pred,
                     'Y_true': y_test_true},
                    open(os.path.join(out_dir,model_name_prefix + '_corrected_results_onTestSet_best_checkpoint.pckl'), 'wb'))

print('ROC-AUC on val dataset: {}'.format(roc_auc_score(y_val_true, y_val_pred)))
print('ROC-AUC on test dataset: {}'.format(roc_auc_score(y_test_true, y_test_pred)))
