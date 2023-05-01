#!/usr/bin/env python3
import networkx as nx
import logging
import pickle
import collections
import multiprocessing as mp
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader



class MIFDataset(Dataset):
    def __init__(self, prot_ids, labels, feat_dir='/home/wjin/data2/protein_structures/MIF-ST_input/', transform=None, target_transform=None):
        self.prot_ids=prot_ids
        self.feat_dir = feat_dir
        self.transform = transform
        self.labels=labels
        assert len(self.prot_ids) == len(self.labels)
        self.label_dic = {idx:lab for idx, lab in zip(self.prot_ids, self.labels)}
        self.target_transform = target_transform
        #self.__indices__=self.prot_names
        super(MIFDataset, self).__init__()
        
    def __len__(self):
        return len(self.prot_ids)

    def __getitem__(self, idx):
        prot_id=self.prot_ids[idx]
        logging.info(prot_id) 
        data_file=os.path.join(self.feat_dir,"AF-{}-model_v2_MIF-ST_input.npz".format(prot_id))
        if not os.path.exists(data_file):
            raise ValueError('{} is not found in dataset dirctory.'.format(prot_id))
        data = np.load(data_file)
        src, edges, nodes, connections, edge_mask = data.values()
        #label = np.array([1] if idx.split('-')[0] in self.RBP_set else [0])
        label = np.array([self.label_dic[prot_id]])
        return torch.from_numpy(src), torch.from_numpy(nodes), torch.from_numpy(edges), torch.from_numpy(connections), torch.from_numpy(edge_mask), torch.from_numpy(label), prot_id  
    
class Protein_Structures_PyG(Dataset):
    """
    Node attr: (gene_idx, )
    Edge attr: From the contact map.
    """
    def __init__(self, root, list_IDs, labels, embed_dir='/home/wjin/data2/protein_structures/AA_Embedding/DeepFRI_emb',embed_name=None, use_distance=True, use_pLDDT=False, transform = None, pre_transform=None):
        """
        gene_kept_ratio: float, [0,1]. The percentage of most highly expressed genes to be used to construct PPI network based on their expression. E.g. gene_kept_ratio=0.5 means we use the top 50% highly expressed genes to construct the PPI network.
        GeneID_idx_dic: the dictionary storing the mapping between UniprotID with the index of the gene in embedding weight matrix.
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.label_dic = {idx:lab for idx, lab in zip(self.list_IDs, self.labels)}
        self.prot_names=list_IDs
        self.use_distance=use_distance
        self.use_pLDDT=use_pLDDT
        self.embed_dir=embed_dir
        self.embed_name=embed_name
        self.AAs=['ALA',
                 'ARG',
                 'ASN',
                 'ASP',
                 'CYS',
                 'GLN',
                 'GLU',
                 'GLY',
                 'HIS',
                 'ILE',
                 'LEU',
                 'LYS',
                 'MET',
                 'PHE',
                 'PRO',
                 'SER',
                 'THR',
                 'TRP',
                 'TYR',
                 'VAL']
        ident=np.identity(len(self.AAs))
        self.aa_vec={self.AAs[i]: ident[i] for i in range(len(self.AAs))}
        self.lfunc = lambda e: int(float(e))
        self.root=root
        if self.embed_name=='AAindex':
            self.aa_to_idx=np.load(os.path.join(self.embed_dir,'AA_index_embedding_weight_matrix.npz'),allow_pickle=True)['aa_to_idx'].item()
        super(Protein_Structures_PyG, self).__init__(root, transform, pre_transform)
        self.__indices__=self.prot_names
        
            
    @property
    def raw_file_names(self):
        return ['AF-{}-model_v2._coords.csv'.format(prot_name) for prot_name in self.prot_names]
        #return ['Expression_data/COADREAD_RNAseq_expression_TPM.csv', 'Class_labels/COADREAD_CMS_labels.csv']
        #return ['TRM_node_attributes_final/'+sample_name+'_node_features.txt' for sample_name in self.sample_names]
    
    @property
    def processed_file_names(self):
        if self.embed_name:
            if self.use_pLDDT:
                if self.use_distance:
                    return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_EdgeDistPLDDT.pt'.format(ID,self.embed_name)) for ID in self.prot_names]
                else:
                    return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_EdgePLDDTonly.pt'.format(ID,self.embed_name)) for ID in self.prot_names]
            else:
                return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed.pt'.format(ID,self.embed_name)) for ID in self.prot_names]
        else:
            if self.use_pLDDT:
                if self.use_distance:
                    return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_EdgeDistPLDDT.pt'.format(ID,self.embed_name)) for ID in self.prot_names]
                else:
                    return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_EdgePLDDTonly.pt'.format(ID,self.embed_name)) for ID in self.prot_names]
            else:
                return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah.pt'.format(ID,self.embed_name)) for ID in self.prot_names]
        
    def download(self):
        # Download to `self.raw_dir`.
        print('To be implemented.')

    def prcess_func(self, prot_name):
        ID = prot_name
        # Load data and get label
        print(ID)
        coord_f=os.path.join(self.raw_dir,'AF-'+ID+'-model_v2._coords.csv')
        pLDDT_f=os.path.join(self.raw_dir,'AF-'+ID+'-model_v2._pLDDT.csv')
        graph_f=os.path.join(self.raw_dir,'AF-'+ID+'-model_v2.pdb_edgelistWithWeight.csv')
        if self.embed_name:
            if self.embed_name=='ProteinBERT-RBP-trainingSet':
                n='ProteinBERT-RBP'
            else:
                n=self.embed_name
            node_f=os.path.join(self.embed_dir,'AF-'+ID+'-model_v2.'+n+'_embedding.npz')
        
        G=nx.read_edgelist(graph_f,delimiter=',',data=[('weight',float)])
        G.remove_edges_from(nx.selfloop_edges(G))

        coords_df=pd.read_csv(coord_f, header=None)
        assert len(coords_df)==G.number_of_nodes(), "Length inconsistency found between network and coordinate files."

        
        #node_feat=np.stack(list(coords_df[0].apply(lambda x: self.aa_vec[x])),axis=0)
        coords=np.array(coords_df[[1,2,3]])
        if self.embed_name == 'AAindex':
            node_feat=np.array([self.aa_to_idx[seq1(x)] for x in list(coords_df[0])])
        elif self.embed_name:
            try:
                node_feat=np.load(node_f)['embedding']
            except:
                print('The embedding file of {} is missing.'.format(ID))
                return
        else:
            node_feat=np.stack(list(coords_df[0].apply(lambda x: self.aa_vec[x])),axis=0)
        edges_all=pd.DataFrame(list(G.edges(data=True)))
        #print(edges_all.shape)
        #print(edges_all)
        edges=np.array(edges_all[[0,1]].applymap(lambda x: int(float(x))))
        if self.use_pLDDT:
            pLDDT_dic=pd.read_csv(pLDDT_f, header=None).to_dict()[1]
            pLDDTs=[]
            for a, b in edges:
                if abs(a-b)<=1: ## If the two amino acids are adjacent in the sequence, their probablity of connection is 100%.
                    pLDDTs.append(1.0)
                else:
                    pLDDTs.append(np.mean([pLDDT_dic[a],pLDDT_dic[b]])*0.01) # average the pLDDT score of the two nodes, and scale it to [0,1].
            pLDDTs=np.array(pLDDTs)
        edges=np.stack((edges[:,0], edges[:,1]),axis=0)
        if node_feat.shape[0] != (edges.max()+1):
            print('The embedding and network files of {} are not agreed by each other.'.format(ID))
            return
        
        if self.use_distance:
            edges_weight=np.array(list(map(lambda x: x['weight'], edges_all[2])))
            if self.use_pLDDT:
                edges_attr=np.stack((edges_weight,pLDDTs),axis=1)
            else:
                edges_attr=edges_weight
        else:
            if self.use_pLDDT:
                edges_attr=pLDDTs
        y = np.array([self.label_dic[ID]])
        
        data = Data(x=torch.from_numpy(node_feat), edge_index=torch.from_numpy(edges), edge_attr=torch.from_numpy(edges_attr), y=torch.from_numpy(y), pos=torch.from_numpy(coords))
        if self.embed_name:
            if self.use_pLDDT:
                if self.use_distance:
                    torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_EdgeDistPLDDT.pt'.format(ID,self.embed_name)))
                else:
                    torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_EdgePLDDTonly.pt'.format(ID,self.embed_name)))
            else:
                torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed.pt'.format(ID,self.embed_name)))
        else:
            if self.use_pLDDT:
                if self.use_distance:
                    torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_EdgeDistPLDDT.pt'.format(ID,self.embed_name)))
                else:
                    torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_EdgePLDDTonly.pt'.format(ID,self.embed_name)))
            else:
                torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah.pt'.format(ID,self.embed_name)))
    def process(self):
        if self.embed_name:
            if self.use_pLDDT:
                if self.use_distance:
                    fn='AF-{}-model_v2_ContactMapGrpah_{}Embed_EdgeDistPLDDT.pt'
                else:
                    fn='AF-{}-model_v2_ContactMapGrpah_{}Embed_EdgePLDDTonly.pt'
            else:
                fn='AF-{}-model_v2_ContactMapGrpah_{}Embed.pt'
        else:
            if self.use_pLDDT:
                if self.use_distance:
                    fn='AF-{}-model_v2_ContactMapGrpah_EdgeDistPLDDT.pt'
                else:
                    fn='AF-{}-model_v2_ContactMapGrpah_EdgePLDDTonly.pt'
            else:
                fn='AF-{}-model_v2_ContactMapGrpah.pt'
                
        if self.embed_name:
            pa=os.path.join(self.processed_dir, fn.format(prot_name,self.embed_name))
        else:
            pa=os.path.join(self.processed_dir, fn.format(prot_name))
        for prot_name in self.prot_names:
            if os.path.exists(pa):
                return
            else:
                self.prcess_func(prot_name)
                
#         pool=mp.Pool(processes=None)
#         results=[pool.apply_async(self.prcess_func, args=(prot_name,)) for prot_name in self.prot_names]
#         results=[p.get() for p in results]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        prot_name=self.prot_names[idx]
        logging.info(prot_name) 
        if self.embed_name:
            if self.use_pLDDT:
                if self.use_distance:
                    data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_EdgeDistPLDDT.pt'.format(prot_name,self.embed_name)))
                else:
                    data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_EdgePLDDTonly.pt'.format(prot_name,self.embed_name)))
            else:
                data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed.pt'.format(prot_name,self.embed_name)))
        else:
            if self.use_pLDDT:
                if self.use_distance:
                    data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_EdgeDistPLDDT.pt'.format(prot_name,self.embed_name)))
                else:
                    data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_EdgePLDDTonly.pt'.format(prot_name,self.embed_name)))
            else:
                data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah.pt'.format(prot_name)))
        
        #print(data.y, prot_name)
        data.prot_name=prot_name
        return data


class Protein_Structures_PyG2(Dataset):
    """
    v2: combine two embeddings 
    Node attr: (gene_idx, )
    Edge attr: From the contact map.
    """
    def __init__(self, root, list_IDs, labels, embed_dir1='/home/wjin/data2/protein_structures/AA_Embedding/DeepFRI_emb',embed_dir2=None, embed_name1='DeepFRI', embed_name2=None, use_distance=True, use_pLDDT=False, transform = None, pre_transform=None):
        """
        gene_kept_ratio: float, [0,1]. The percentage of most highly expressed genes to be used to construct PPI network based on their expression. E.g. gene_kept_ratio=0.5 means we use the top 50% highly expressed genes to construct the PPI network.
        GeneID_idx_dic: the dictionary storing the mapping between UniprotID with the index of the gene in embedding weight matrix.
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.label_dic = {idx:lab for idx, lab in zip(self.list_IDs, self.labels)}
        self.prot_names=list_IDs
        self.use_distance=use_distance
        self.use_pLDDT=use_pLDDT
        self.embed_dir1=embed_dir1
        self.embed_dir2=embed_dir2
        self.embed_name1=embed_name1
        self.embed_name2=embed_name2
        if self.embed_name1 == None:
            self.embed_name1 =='OneHot'
        if self.embed_name2 == None:
            self.embed_name2 =='OneHot'
        self.AAs=['ALA',
                 'ARG',
                 'ASN',
                 'ASP',
                 'CYS',
                 'GLN',
                 'GLU',
                 'GLY',
                 'HIS',
                 'ILE',
                 'LEU',
                 'LYS',
                 'MET',
                 'PHE',
                 'PRO',
                 'SER',
                 'THR',
                 'TRP',
                 'TYR',
                 'VAL']
        ident=np.identity(len(self.AAs))
        self.aa_vec={self.AAs[i]: ident[i] for i in range(len(self.AAs))}
        self.lfunc = lambda e: int(float(e))
        self.root=root
        super(Protein_Structures_PyG2, self).__init__(root, transform, pre_transform)
        self.__indices__=self.prot_names
        
            
    @property
    def raw_file_names(self):
        return ['AF-{}-model_v2._coords.csv'.format(prot_name) for prot_name in self.prot_names]
        #return ['Expression_data/COADREAD_RNAseq_expression_TPM.csv', 'Class_labels/COADREAD_CMS_labels.csv']
        #return ['TRM_node_attributes_final/'+sample_name+'_node_features.txt' for sample_name in self.sample_names]
    
    @property
    def processed_file_names(self):
        if self.use_pLDDT:
            if self.use_distance:
                return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed_EdgeDistPLDDT.pt'.format(ID,self.embed_name1,self.embed_name2)) for ID in self.prot_names]
            else:
                return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed_EdgePLDDTonly.pt'.format(ID,self.embed_name1,self.embed_name2)) for ID in self.prot_names]
        else:
            return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed.pt'.format(ID,self.embed_name1,self.embed_name2)) for ID in self.prot_names]
    
    def download(self):
        # Download to `self.raw_dir`.
        print('To be implemented.')

    def prcess_func(self, prot_name):
        ID = prot_name
        # Load data and get label
        print(ID)
        coord_f=os.path.join(self.raw_dir,'AF-'+ID+'-model_v2._coords.csv')
        pLDDT_f=os.path.join(self.raw_dir,'AF-'+ID+'-model_v2._pLDDT.csv')
        graph_f=os.path.join(self.raw_dir,'AF-'+ID+'-model_v2.pdb_edgelistWithWeight.csv')
        if (not self.embed_name1==None) and (not self.embed_name1=='OneHot'):
            node_f1=os.path.join(self.embed_dir1,'AF-'+ID+'-model_v2.'+self.embed_name1+'_embedding.npz')
        if (not self.embed_name2==None) and (not self.embed_name2=='OneHot'):
            if self.embed_name2=='Phi_Psi':
                node_f2=os.path.join(self.embed_dir2,'AF-'+ID+'-model_v2_phi_psi.npz')
            else:
                node_f2=os.path.join(self.embed_dir2,'AF-'+ID+'-model_v2.'+self.embed_name2+'_embedding.npz')

        G=nx.read_edgelist(graph_f,delimiter=',',data=[('weight',float)])
        G.remove_edges_from(nx.selfloop_edges(G))

        coords_df=pd.read_csv(coord_f, header=None)
        assert len(coords_df)==G.number_of_nodes(), "Length inconsistency found between network and coordinate files."

        
        #node_feat=np.stack(list(coords_df[0].apply(lambda x: self.aa_vec[x])),axis=0)
        coords=np.array(coords_df[[1,2,3]])
        if self.embed_name1!='OneHot':
            try:
                node_feat1=np.load(node_f1)['embedding']
            except:
                print('The embedding file1 of {} is missing.'.format(ID))
                return
        else:
            node_feat1=np.stack(list(coords_df[0].apply(lambda x: self.aa_vec[x])),axis=0)
            
        if self.embed_name2=='Phi_Psi':
            try:
                d=np.load(node_f2,allow_pickle=True)
                angle1=np.array(list(map(lambda x: x if x != None else 0, d['phi'])))
                angle2=np.array(list(map(lambda x: x if x != None else 0, d['psi'])))
                node_feat2=np.stack([angle1,angle2],axis=1)
            except:
                print('The embedding file2 of {} is missing.'.format(ID))
                return
        elif self.embed_name2!='OneHot':
            try:
                node_feat2=np.load(node_f2)['embedding']
            except:
                print('The embedding file of {} is missing.'.format(ID))
                return
        else:
            node_feat2=np.stack(list(coords_df[0].apply(lambda x: self.aa_vec[x])),axis=0)
        
        try:
            node_feat=np.concatenate([node_feat1, node_feat2],axis=1)    
        except ValueError:
            print('{} has conflicts on the node feature dimensions.'.format(ID))
            return
        edges_all=pd.DataFrame(list(G.edges(data=True)))
        #print(edges_all.shape)
        #print(edges_all)
        edges=np.array(edges_all[[0,1]].applymap(lambda x: int(float(x))))
        if self.use_pLDDT:
            pLDDT_dic=pd.read_csv(pLDDT_f, header=None).to_dict()[1]
            pLDDTs=[]
            for a, b in edges:
                if abs(a-b)<=1: ## If the two amino acids are adjacent in the sequence, their probablity of connection is 100%.
                    pLDDTs.append(1.0)
                else:
                    pLDDTs.append(np.mean([pLDDT_dic[a],pLDDT_dic[b]])*0.01) # average the pLDDT score of the two nodes, and scale it to [0,1].
            pLDDTs=np.array(pLDDTs)
        edges=np.stack((edges[:,0], edges[:,1]),axis=0)
        if node_feat.shape[0] != (edges.max()+1):
            print('The embedding and network files of {} are not agreed by each other.'.format(ID))
            return             
        if self.use_distance:
            edges_weight=np.array(list(map(lambda x: x['weight'], edges_all[2])))
            if self.use_pLDDT:
                edges_attr=np.stack((edges_weight,pLDDTs),axis=1)
            else:
                edges_attr=edges_weight
        else:
            if self.use_pLDDT:
                edges_attr=pLDDTs
        
        y = np.array([self.label_dic[ID]])
        
        data = Data(x=torch.from_numpy(node_feat), edge_index=torch.from_numpy(edges), edge_attr=torch.from_numpy(edges_attr), y=torch.from_numpy(y), pos=torch.from_numpy(coords))
        if self.use_pLDDT:
            if self.use_distance:
                torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed_EdgeDistPLDDT.pt'.format(ID,self.embed_name1,self.embed_name2)))
            else:
                torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed_EdgePLDDTonly.pt'.format(ID,self.embed_name1,self.embed_name2)))
        else:
            torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed.pt'.format(ID,self.embed_name1,self.embed_name2)))
            
    def process(self):
        if self.use_pLDDT:
            if self.use_distance:
                fn='AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed_EdgeDistPLDDT.pt'
            else:
                fn='AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed_EdgePLDDTonly.pt'
        else:
            fn='AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed.pt'
        for prot_name in self.prot_names:
            pa=os.path.join(self.processed_dir, fn.format(prot_name,self.embed_name1,self.embed_name2))
            if os.path.exists(pa):
                return
            else:
                self.prcess_func(prot_name)
        
#         pool=mp.Pool(processes=None)
#         results=[pool.apply_async(self.prcess_func, args=(prot_name,)) for prot_name in self.prot_names]
#         results=[p.get() for p in results]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        prot_name=self.prot_names[idx]
        logging.info(prot_name) 
        if self.embed_name1==None:
            self.embed_name1=='OneHot'
        if self.embed_name2==None:
            self.embed_name2=='OneHot'
            
        if self.use_pLDDT:
            if self.use_distance:
                data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed_EdgeDistPLDDT.pt'.format(prot_name,self.embed_name1,self.embed_name2)))
            else:
                data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed_EdgePLDDTonly.pt'.format(prot_name,self.embed_name1,self.embed_name2)))
        else:
            data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}-{}-Embed.pt'.format(prot_name,self.embed_name1,self.embed_name2)))
        #print(data.y, prot_name)
        data.prot_name=prot_name
        return data


class Protein_Structures_PyG_edges(Dataset):
    """
    this version: add complicated edge attributes. E.g. 3Di for each pair of interacted nodes.
    Node attr: (gene_idx, )
    Edge attr: From .
    """
    def __init__(self, root, list_IDs, labels, embed_dir='/home/wjin/data2/protein_structures/AA_Embedding/DeepFRI_emb',embed_name=None, use_pLDDT=False, transform = None, pre_transform=None):
        """
        gene_kept_ratio: float, [0,1]. The percentage of most highly expressed genes to be used to construct PPI network based on their expression. E.g. gene_kept_ratio=0.5 means we use the top 50% highly expressed genes to construct the PPI network.
        GeneID_idx_dic: the dictionary storing the mapping between UniprotID with the index of the gene in embedding weight matrix.
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.label_dic = {idx:lab for idx, lab in zip(self.list_IDs, self.labels)}
        self.prot_names=list_IDs
        self.use_pLDDT=use_pLDDT
        self.embed_dir=embed_dir
        self.embed_name=embed_name
        self.AAs=['ALA',
                 'ARG',
                 'ASN',
                 'ASP',
                 'CYS',
                 'GLN',
                 'GLU',
                 'GLY',
                 'HIS',
                 'ILE',
                 'LEU',
                 'LYS',
                 'MET',
                 'PHE',
                 'PRO',
                 'SER',
                 'THR',
                 'TRP',
                 'TYR',
                 'VAL']
        ident=np.identity(len(self.AAs))
        self.aa_vec={self.AAs[i]: ident[i] for i in range(len(self.AAs))}
        self.lfunc = lambda e: int(float(e))
        self.root=root
        if self.embed_name=='AAindex':
            self.aa_to_idx=np.load(os.path.join(self.embed_dir,'AA_index_embedding_weight_matrix.npz'),allow_pickle=True)['aa_to_idx'].item()
        super(Protein_Structures_PyG_edges, self).__init__(root, transform, pre_transform)
        self.__indices__=self.prot_names
        
            
    @property
    def raw_file_names(self):
        return ['AF-{}-model_v2._coords.csv'.format(prot_name) for prot_name in self.prot_names]
        #return ['Expression_data/COADREAD_RNAseq_expression_TPM.csv', 'Class_labels/COADREAD_CMS_labels.csv']
        #return ['TRM_node_attributes_final/'+sample_name+'_node_features.txt' for sample_name in self.sample_names]
    
    @property
    def processed_file_names(self):
        if self.embed_name:
            if self.use_pLDDT:
                return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_Edge3Di_pLDDT.pt'.format(ID,self.embed_name)) for ID in self.prot_names]
            else:
                return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_Edge3Di.pt'.format(ID,self.embed_name)) for ID in self.prot_names]
        else:
            if self.use_pLDDT:
                return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_Edge3Di_pLDDT.pt'.format(ID,self.embed_name)) for ID in self.prot_names]
            else:
                return [os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_Edge3Di.pt'.format(ID,self.embed_name)) for ID in self.prot_names]
    def download(self):
        # Download to `self.raw_dir`.
        print('To be implemented.')

    def prcess_func(self, prot_name):
        ID = prot_name
        # Load data and get label
        print(ID)
        coord_f=os.path.join(self.raw_dir,'AF-'+ID+'-model_v2._coords.csv')
        pLDDT_f=os.path.join(self.raw_dir,'AF-'+ID+'-model_v2._pLDDT.csv')
        graph_f=os.path.join(self.raw_dir,'AF-'+ID+'-model_v2.pdb_edgelistWithWeight_3Di.csv')
        if self.embed_name:
            if self.embed_name=='ProteinBERT-RBP-trainingSet':
                n='ProteinBERT-RBP'
            else:
                n=self.embed_name
            node_f=os.path.join(self.embed_dir,'AF-'+ID+'-model_v2.'+n+'_embedding.npz')
        
        df=pd.read_csv(graph_f,header=None)
        G=nx.from_pandas_edgelist(df,0,1,edge_attr=True)
        #G=nx.read_edgelist(graph_f,delimiter=',',data=[('weight',float)])
        G.remove_edges_from(nx.selfloop_edges(G))

        coords_df=pd.read_csv(coord_f, header=None)
        assert len(coords_df)==G.number_of_nodes(), "Length inconsistency found between network and coordinate files."

        
        #node_feat=np.stack(list(coords_df[0].apply(lambda x: self.aa_vec[x])),axis=0)
        coords=np.array(coords_df[[1,2,3]])
        if self.embed_name == 'AAindex':
            node_feat=np.array([self.aa_to_idx[seq1(x)] for x in list(coords_df[0])])
        elif self.embed_name:
            try:
                node_feat=np.load(node_f)['embedding']
            except:
                print('The embedding file of {} is missing.'.format(ID))
                return
        else:
            node_feat=np.stack(list(coords_df[0].apply(lambda x: self.aa_vec[x])),axis=0)
        edges_all=pd.DataFrame(list(G.edges(data=True)))
        #print(edges_all.shape)
        #print(edges_all)
        edges=np.array(edges_all[[0,1]].applymap(lambda x: int(float(x))))
        if self.use_pLDDT:
            pLDDT_dic=pd.read_csv(pLDDT_f, header=None).to_dict()[1]
            pLDDTs=[]
            for a, b in edges:
                if abs(a-b)<=1: ## If the two amino acids are adjacent in the sequence, their probablity of connection is 100%.
                    pLDDTs.append(1.0)
                else:
                    pLDDTs.append(np.mean([pLDDT_dic[a],pLDDT_dic[b]])*0.01) # average the pLDDT score of the two nodes, and scale it to [0,1].
            pLDDTs=np.array(pLDDTs)
        edges=np.stack((edges[:,0], edges[:,1]),axis=0)
        if node_feat.shape[0] != (edges.max()+1):
            print('The embedding and network files of {} are not agreed by each other.'.format(ID))
            return             
        edges_attr=np.array(list(map(lambda x: list(collections.OrderedDict(sorted(x.items())).values()), edges_all[2])))
        if self.use_pLDDT:
            edges_attr=np.concatenate([edges_attr,pLDDTs[:,None]],axis=1)
        y = np.array([self.label_dic[ID]])
        
        data = Data(x=torch.from_numpy(node_feat), edge_index=torch.from_numpy(edges), edge_attr=torch.from_numpy(edges_attr), y=torch.from_numpy(y), pos=torch.from_numpy(coords))
        if self.embed_name:
            if self.use_pLDDT:
                torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_Edge3Di_pLDDT.pt'.format(ID,self.embed_name)))
            else:
                torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_Edge3Di.pt'.format(ID,self.embed_name)))
        else:
            if self.use_pLDDT:
                torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_Edge3Di_pLDDT.pt'.format(ID,self.embed_name)))
            else:
                torch.save(data, os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_Edge3Di.pt'.format(ID,self.embed_name)))

    def process(self):
        if self.embed_name:
            if self.use_pLDDT:
                fn='AF-{}-model_v2_ContactMapGrpah_{}Embed_Edge3Di_pLDDT.pt'
            else:
                fn='AF-{}-model_v2_ContactMapGrpah_{}Embed_Edge3Di.pt'
        else:
            if self.use_pLDDT:
                fn='AF-{}-model_v2_ContactMapGrpah_Edge3Di_pLDDT.pt'
            else:
                fn='AF-{}-model_v2_ContactMapGrpah_Edge3Di.pt'

        
        if self.embed_name:
            pa=os.path.join(self.processed_dir, fn.format(prot_name,self.embed_name))
        else:
            pa=os.path.join(self.processed_dir, fn.format(prot_name))
        for prot_name in self.prot_names:
            if os.path.exists(pa):
                return
            else:
                self.prcess_func(prot_name)
        
#         pool=mp.Pool(processes=None)
#         results=[pool.apply_async(self.prcess_func, args=(prot_name,)) for prot_name in self.prot_names]
#         results=[p.get() for p in results]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        prot_name=self.prot_names[idx]
        logging.info(prot_name) 
        if self.embed_name:
            if self.use_pLDDT:
                data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_Edge3Di_pLDDT.pt'.format(prot_name,self.embed_name)))
            else:
                data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_{}Embed_Edge3Di.pt'.format(prot_name,self.embed_name)))
        else:
            if self.use_pLDDT:
                data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_Edge3Di_pLDDT.pt'.format(prot_name,self.embed_name)))
            else:
                data = torch.load(os.path.join(self.processed_dir, 'AF-{}-model_v2_ContactMapGrpah_Edge3Di.pt'.format(prot_name,self.embed_name)))
        
        #print(data.y, prot_name)
        data.prot_name=prot_name
        return data
