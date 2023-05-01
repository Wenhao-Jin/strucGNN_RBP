import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, SAGEConv

import torch
# from egnn_pytorch_lucidrains.egnn_pytorch import EGNN_Network, EGNN
# from egnn_pytorch_lucidrains.egnn_pytorch.egnn_pytorch_geometric import EGNN_Sparse_Network, EGNN_Sparse

class GNN_modules(nn.Module):
    def __init__(self, n_layers, feats_dim):
        super().__init__()
        #self.egnn_blocks=nn.Sequential(*[EGNN(in_node_nf=in_nf, hidden_nf=out_nf, out_node_nf=1, in_edge_nf=1) for in_nf, out_nf in zip(hidden_nfs, hidden_nfs[1:])])
        self.gnn_blocks=EGNN_Sparse_Network(n_layers=n_layers, feats_dim=feats_dim)
        
        Sequential('x, edge_index, batch', [
                (Dropout(p=0.5), 'x -> x'),
                (GCNConv(dataset.num_features, 64), 'x, edge_index -> x1'),
                ReLU(inplace=True),
                (GCNConv(64, 64), 'x1, edge_index -> x2'),
                ReLU(inplace=True),
                (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
                (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
                (global_mean_pool, 'x, batch -> x'),
                Linear(2 * 64, dataset.num_classes),
            ])

    def forward(self, x, pos, edges, edge_attr):
        #return self.egnn_blocks(h, x, edges, edge_attr)
        return self.egnn_blocks(torch.concat([pos,x],dim=1), edges, edge_attr)

class GNN_modules_concat(nn.Module):
    def __init__(self, in_node_channel, hidden_nfs, gnn_layer='GCNConv',has_bias=True, heads=1, edge_dim=None, deg=None):
        super().__init__()
        self.hidden_nfs=[in_node_channel]+hidden_nfs
        self.gnn_layer=gnn_layer
        self.has_bias=has_bias
        if self.gnn_layer=='GCNConv':
            self.convs=nn.ModuleList([GCNConv(in_channels=in_nf, out_channels=out_nf, bias=has_bias) for in_nf, out_nf in zip(self.hidden_nfs, self.hidden_nfs[1:])])
        elif self.gnn_layer=='GATv2Conv':
            self.heads=heads
            self.edge_dim=edge_dim
            self.convs=nn.ModuleList([GATv2Conv(in_channels=in_nf, out_channels=out_nf, heads=heads, edge_dim=self.edge_dim, bias=has_bias) for in_nf, out_nf in zip(self.hidden_nfs, self.hidden_nfs[1:])])
        elif self.gnn_layer=='NNConv':
            self.edge_dim=edge_dim
            #self.convs=nn.ModuleList([NNConv(in_channels=in_nf, out_channels=out_nf,nn=torch.nn.Linear(edge_di, in_nf*out_nf), bias=has_bias) for in_nf, out_nf, edge_di in zip(self.hidden_nfs, self.hidden_nfs[1:], [self.edge_dim]+[i*j for i,j in zip(self.hidden_nfs[:-1], self.hidden_nfs[1:])])])
            self.convs=nn.ModuleList([NNConv(in_channels=in_nf, out_channels=out_nf,nn=torch.nn.Linear(self.edge_dim, in_nf*out_nf), bias=has_bias) for in_nf, out_nf in zip(self.hidden_nfs, self.hidden_nfs[1:])])
        elif self.gnn_layer=='PNAConv':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            self.edge_dim=edge_dim
            self.deg=deg
            self.convs=nn.ModuleList([PNAConv(in_channels=in_nf, out_channels=out_nf,
                           aggregators=aggregators, scalers=scalers, deg=self.deg,
                           edge_dim=self.edge_dim) for in_nf, out_nf in zip(self.hidden_nfs, self.hidden_nfs[1:])])
        elif self.gnn_layer=='GeneralConv':
            self.heads=heads
            self.edge_dim=edge_dim
            self.convs=nn.ModuleList([GeneralConv(in_channels=in_nf, out_channels=out_nf, heads=heads, in_edge_channels=self.edge_dim, bias=has_bias) for in_nf, out_nf in zip(self.hidden_nfs, self.hidden_nfs[1:])])
        elif self.gnn_layer=='SplineConv':
            self.heads=heads
            self.edge_dim=edge_dim
            self.convs=nn.ModuleList([SplineConv(in_channels=in_nf, out_channels=out_nf, heads=heads, in_edge_channels=self.edge_dim, bias=has_bias) for in_nf, out_nf in zip(self.hidden_nfs, self.hidden_nfs[1:])])
        
        elif self.gnn_layer=='AttentiveFP':
            self.edge_dim=edge_dim
            self.convs=AttentiveFP(in_channels=self.hidden_nfs[0], hidden_channels=self.hidden_nfs[1], out_channels=self.hidden_nfs[-1], num_layers=len(self.hidden_nfs)-1, num_timesteps=2, edge_dim=self.edge_dim)
        self.batch_norms=nn.ModuleList([BatchNorm(hidden_nf) for hidden_nf in hidden_nfs])
        
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        h_list=[]
        if self.gnn_layer=='AttentiveFP':
            #print(x.shape, edge_index.shape, edge_attr.shape)
            out=self.convs(x, edge_index, edge_attr, batch)
            return out
        else:
            for conv, batch_norm in zip(self.convs, self.batch_norms):
                if self.gnn_layer=='GCNConv' or self.gnn_layer=='GATv2Conv' or self.gnn_layer=='GeneralConv' or self.gnn_layer=='PNAConv' or self.gnn_layer=='NNConv':
                    #print(x.shape, edge_index.shape, edge_attr.shape)
                    x = conv(x, edge_index,edge_attr)
                else:
                    x = conv(x, edge_index,batch,edge_attr)
                #x = batch_norm(x)
                x = F.elu(x)
                #x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
                h_list.append(x)

            return torch.cat(h_list, 1)

def MLP(in_feat, out_feat, has_bias=True):
    return Sequential(Linear(in_feat, out_feat, bias=has_bias), ReLU(), BatchNorm1d(out_feat))

class MLP_modules(nn.Module):
    def __init__(self, hidden_feats, has_bias=True):
        super().__init__()
        self.MLP_blocks=Sequential(*[MLP(in_feat,out_feat,has_bias=has_bias) for in_feat, out_feat in zip(hidden_feats, hidden_feats[1:])])
    def forward(self, x):
        return self.MLP_blocks(x)
        
    
class GNN_Net0(nn.Module):
    def __init__(self, in_node_channel, hidden_nfs, hidden_fc_feats, gnn_layer='GCNConv', globalpool=global_max_pool,output_channel=1, has_gnn_bias=True, has_fc_bias=True, heads=1, edge_dim=None, deg=None, **kwargs):
        super().__init__()
        self.GNNs=GNN_modules_concat(in_node_channel, hidden_nfs, gnn_layer, has_bias=has_gnn_bias, heads=heads, edge_dim=edge_dim,deg=deg)
        self.convs=self.GNNs.convs
        self.globalpool=globalpool
        self.hidden_fc_feats=[sum(hidden_nfs)]+hidden_fc_feats
        #self.fc=MLP_modules(self.hidden_fc_feats, has_bias=has_fc_bias)
        #self.fc=Sequential(*[MLP(in_feat,out_feat,has_bias=has_fc_bias) for in_feat, out_feat in zip(hidden_fc_feats, hidden_fc_feats[1:])])
        self.fc=Sequential(Linear(self.hidden_fc_feats[0], self.hidden_fc_feats[1], bias=has_fc_bias), ReLU(), BatchNorm1d(self.hidden_fc_feats[1]))
        self.has_gnn_bias=has_gnn_bias
        self.has_fc_bias=has_fc_bias
        self.gnn_layer=gnn_layer
        self.edge_dim=edge_dim
        self.heads=heads
        self.deg=deg
        if output_channel==1:
            self.last_layer=Sequential(nn.Linear(self.hidden_fc_feats[-1] , output_channel, bias=has_fc_bias), Sigmoid())
        elif output_channel>1:
            self.last_layer=Sequential(nn.Linear(self.hidden_fc_feats[-1] , output_channel, bias=has_fc_bias), LogSoftmax())
        else:
            raise ValueError('output_channel should be positive integer.')
    def forward(self, x, edges, batch=None, edge_attr=None):
        
        if self.gnn_layer=='AttentiveFP':
            out=self.GNNs(x,edges,batch,edge_attr)
            s = nn.Sigmoid()
            return s(out)
        else:
            #print(edges.shape,edge_attr.shape)
            out=self.GNNs(x,edges,batch,edge_attr)
            # out=out[:, pos_dim:] ## Only keep the node features
            #print(out.shape,batch.shape)
            out=self.globalpool(out, batch=batch)
            out=self.fc(out)
            out=self.last_layer(out)
            #print(out.shape)
            return out
     
def create_emb_layer(weights_matrix, trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if trainable:
        emb_layer.weight.requires_grad = True
    else:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim
        
class GNN_Net0_AAindex(nn.Module):
    def __init__(self, in_node_channel, hidden_nfs, hidden_fc_feats, AAindex_matrix, gnn_layer='GCNConv', globalpool=global_max_pool,output_channel=1, has_gnn_bias=True, has_fc_bias=True, **kwargs):
        super().__init__()
        self.AAindex_matrix=AAindex_matrix
        self.embed, self.num_embeddings, self.embedding_dim=create_emb_layer(torch.from_numpy(self.AAindex_matrix),trainable=False)
        self.GNNs=GNN_modules_concat(in_node_channel, hidden_nfs, gnn_layer, has_bias=has_gnn_bias)
        self.globalpool=globalpool
        self.hidden_fc_feats=[sum(hidden_nfs)]+hidden_fc_feats
        #self.fc=MLP_modules(self.hidden_fc_feats, has_bias=has_fc_bias)
        #self.fc=Sequential(*[MLP(in_feat,out_feat,has_bias=has_fc_bias) for in_feat, out_feat in zip(hidden_fc_feats, hidden_fc_feats[1:])])
        self.fc=Sequential(Linear(self.hidden_fc_feats[0], self.hidden_fc_feats[1], bias=has_fc_bias), ReLU(), BatchNorm1d(self.hidden_fc_feats[1]))
        self.has_gnn_bias=has_gnn_bias
        self.has_fc_bias=has_fc_bias
        if output_channel==1:
            self.last_layer=Sequential(nn.Linear(self.hidden_fc_feats[-1] , output_channel, bias=has_fc_bias), Sigmoid())
        elif output_channel>1:
            self.last_layer=Sequential(nn.Linear(self.hidden_fc_feats[-1] , output_channel, bias=has_fc_bias), LogSoftmax())
        else:
            raise ValueError('output_channel should be positive integer.')
    def forward(self, x, edges, batch=None, edge_attr=None):
        
        #print(edges.shape,edge_attr.shape)
        embed=self.embed(x)
        out=self.GNNs(embed,edges,batch,edge_attr)
        # out=out[:, pos_dim:] ## Only keep the node features
        out=self.globalpool(out, batch=batch)
        out=self.fc(out)
        out=self.last_layer(out)
        #print(out.shape)
        return out
    
    
def MLP(in_feat, out_feat):
    return Sequential(Linear(in_feat, out_feat), ReLU(), BatchNorm1d(out_feat))

class MLP_modules(nn.Module):
    def __init__(self, hidden_feats):
        super().__init__()
        self.MLP_blocks=Sequential(*[MLP(in_feat,out_feat) for in_feat, out_feat in zip(hidden_feats, hidden_feats[1:])])
    def forward(self, x):
        return self.MLP_blocks(x)
        
        
        
class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], batch_norm=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], batch_norm=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=1.0, k=5):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels])

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x[id_k_neighbor[1]], id_k_neighbor[0],
                               dim_size=id_clusters.size(0), dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]),
            BN(channels[i]) if batch_norm else Identity(), ReLU())
        for i in range(1, len(channels))
    ])


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], batch_norm=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], batch_norm=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=1.0, k=5):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels])

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x[id_k_neighbor[1]], id_k_neighbor[0],
                               dim_size=id_clusters.size(0), dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]),
            BN(channels[i]) if batch_norm else Identity(), ReLU())
        for i in range(1, len(channels))
    ])


class PointTransformerNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=5):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]])

        self.transformer_input = TransformerBlock(in_channels=dim_model[0],
                                                  out_channels=dim_model[0])
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))

        # class score computation
        self.mlp_output = Seq(Lin(dim_model[-1], 64), ReLU(), Lin(64, 64),
                              ReLU(), Lin(64, out_channels))

    def forward(self, x, pos, edge_index, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        #edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.mlp_output(x)

        return out#.sigmoid()
        