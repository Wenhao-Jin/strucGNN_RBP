#!/usr/bin/env python
import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import sys
from proteinbert import InputEncoder, OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune #, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import tensorflow as tf
import random
import pickle

def get_model(seq_len, proteinBERT_modelfile):
    n_annotations=8943
    with open(proteinBERT_modelfile,'rb') as f:
        pretrained_model_generator=pickle.load(f)
        input_encoder = InputEncoder(n_annotations)
    
    model=pretrained_model_generator.create_model(seq_len=seq_len)#,init_weights=False)
    for layer in model.layers:
        #print(layer.name)
        if layer.name=='seq-merge2-block6':
            emb_layer=layer
        if layer.name=='seq-merge2-norm-block6':
            out_layer=layer
            break
    output_layer=out_layer(emb_layer.output)
    model_new = tf.keras.models.Model(inputs = model.input, outputs = output_layer)
    return model_new, input_encoder


## Generate the embeddings
proteinBERT_modelfile='/home/wjin/projects/RBP_pred/RBP_identification/HydRa2.0/data/ProteinBERT/ProteinBERT_TrainedWithTrainingSet_defaultSetting_ModelFile.pkl'
OUTPUT_TYPE = OutputType(False, 'binary')
UNIQUE_LABELS = [0, 1]
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)
seq_len=512

path = '/home/wjin/data3/AlphaFold2/ContactMaps_DeepFRI/'
files=list(filter(lambda x: x.endswith('.ContactMap_addNorm.npz'), os.listdir(path)))
out_dir='/home/wjin/data2/protein_structures/AA_Embedding/ProteinBERT-RBP_emb_trainingset'
seqs=[]
model_dic={}
#raw_Y=[]
#protein_names=[]

random.shuffle(files)
for f in files:
    out_file=f.replace('ContactMap_addNorm.npz','ProteinBERT-RBP_embedding')
    if os.path.exists(os.path.join(out_dir,out_file)):
        continue
        
    cmap = np.load(path + f)
    seq = str(cmap['seqres'])

    seq_len=(int((len(seq)+2)/512)+1 if (len(seq)+2)%512!=0 else int((len(seq)+2)/512))*512
    if seq_len in model_dic:
        model, input_encoder =model_dic[seq_len]
    else:
        model, input_encoder=get_model(seq_len,proteinBERT_modelfile)
        model_dic[seq_len]=(model, input_encoder)
    
    X = input_encoder.encode_X([seq], seq_len) ## encode_X function will add START and END token to the two ends of the sequence.
    embedding = model.predict(X, batch_size = 1)[0][1:len(seq)+1]
    np.savez_compressed(os.path.join(out_dir, out_file),
                            embedding=embedding,
                            seqres=cmap['seqres'],
                            )

