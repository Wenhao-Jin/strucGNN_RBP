## Reference link for finetune: https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout, Sigmoid, ELU

def global_max_pool(x: Tensor):
    """
    Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/glob/glob.html#global_max_pool
    """
    return x.max(dim=-2, keepdim=x.dim() == 2)[0]


def global_add_pool(x: Tensor):
    """
    Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/glob/glob.html#global_add_pool
    """
    return x.sum(dim=-2, keepdim=x.dim() == 2)

def global_mean_pool(x: Tensor):
    """
    Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/glob/glob.html#global_mean_pool
    """
    return x.mean(dim=-2, keepdim=x.dim() == 2)

class MIF_ST_RBP(nn.Module):
    def __init__(self, mif_st_model, hidden_fc_feats, output_channel, globalpool=global_max_pool, has_fc_bias=True, mif_st_out_dim=256, batchnorm=False):
        super(MIF_ST_RBP, self).__init__()
        # Start with MIF-ST model, swap out the final layer because that's the model we had defined above. 
        model = mif_st_model
        #num_final_in = model.fc.in_features
        #model.fc = nn.Linear(num_final_in, 300)
        
        ## Now that the architecture is defined same as above, let's load the model we would have trained above. 
        #checkpoint = torch.load(MODEL_PATH)
        #model.load_state_dict(checkpoint)
        
        # Let's freeze the same as above. Same code as above without the print statements
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False        
        # Now, let's define new layers that we want to add on top. 
        # Basically, these are just objects we define here. The "adding on top" is defined by the forward()
        # function which decides the flow of the input data into the model.
        
        # NOTE - Even the above model needs to be passed to self.
        #self.MIF_ST = nn.Sequential(*list(model.children()))
        self.MIF_ST = model
        self.globalpool=globalpool
        self.hidden_fc_feats=[mif_st_out_dim]+hidden_fc_feats
        self.batchnorm=batchnorm
        
        fc_list=[]
        for ii in range(1,len(self.hidden_fc_feats)):
            fc_list.append(Linear(self.hidden_fc_feats[ii-1], self.hidden_fc_feats[ii], bias=has_fc_bias))
            fc_list.append(ReLU())
            if self.batchnorm:
                fc_list.append(BatchNorm1d(self.hidden_fc_feats[ii]))
        self.fc=Sequential(*fc_list)  
        #self.fc=Sequential(Linear(self.hidden_fc_feats[0], self.hidden_fc_feats[1], bias=has_fc_bias), ReLU(), BatchNorm1d(self.hidden_fc_feats[1]))
        if output_channel==1:
            self.last_layer=Sequential(nn.Linear(self.hidden_fc_feats[-1] , output_channel, bias=has_fc_bias))#, Sigmoid())
        elif output_channel>1:
            self.last_layer=Sequential(nn.Linear(self.hidden_fc_feats[-1] , output_channel, bias=has_fc_bias), LogSoftmax())
        else:
            raise ValueError('output_channel should be positive integer.')
        
    # The forward function defines the flow of the input data and thus decides which layer/chunk goes on top of what.
    def forward(self, src, nodes, edges, connections, edge_mask):
        out = self.MIF_ST(src, nodes, edges, connections, edge_mask)
        #print(out.shape)
        out=self.globalpool(out)
        #print(out.shape)
        out=self.fc(out)
        #print(out.shape)
        out=self.last_layer(out)
        #print(out.shape)
        return out
    
class MIF_ST_RBP_2(nn.Module):
    ### remove last sigmoid layer to work with the BCEWithLogitsLoss
    def __init__(self, mif_st_model, hidden_fc_feats, output_channel, globalpool=global_max_pool, has_fc_bias=True, mif_st_out_dim=256, batchnorm=False):
        super(MIF_ST_RBP_2, self).__init__()
        # Start with MIF-ST model, swap out the final layer because that's the model we had defined above. 
        model = mif_st_model
        #num_final_in = model.fc.in_features
        #model.fc = nn.Linear(num_final_in, 300)
        
        ## Now that the architecture is defined same as above, let's load the model we would have trained above. 
        #checkpoint = torch.load(MODEL_PATH)
        #model.load_state_dict(checkpoint)
        
        # Let's freeze the same as above. Same code as above without the print statements
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False        
        # Now, let's define new layers that we want to add on top. 
        # Basically, these are just objects we define here. The "adding on top" is defined by the forward()
        # function which decides the flow of the input data into the model.
        
        # NOTE - Even the above model needs to be passed to self.
        #self.MIF_ST = nn.Sequential(*list(model.children()))
        self.MIF_ST = model
        self.globalpool=globalpool
        self.hidden_fc_feats=[mif_st_out_dim]+hidden_fc_feats
        self.batchnorm=batchnorm
        
        fc_list=[]
        for ii in range(1,len(self.hidden_fc_feats)):
            fc_list.append(Linear(self.hidden_fc_feats[ii-1], self.hidden_fc_feats[ii], bias=has_fc_bias))
            fc_list.append(ReLU())
            if self.batchnorm:
                fc_list.append(BatchNorm1d(self.hidden_fc_feats[ii]))
        self.fc=Sequential(*fc_list)  
        #self.fc=Sequential(Linear(self.hidden_fc_feats[0], self.hidden_fc_feats[1], bias=has_fc_bias), ReLU(), BatchNorm1d(self.hidden_fc_feats[1]))
        if output_channel==1:
            self.last_layer=Sequential(nn.Linear(self.hidden_fc_feats[-1] , output_channel, bias=has_fc_bias))#, Sigmoid())
        elif output_channel>1:
            self.last_layer=Sequential(nn.Linear(self.hidden_fc_feats[-1] , output_channel, bias=has_fc_bias), LogSoftmax())
        else:
            raise ValueError('output_channel should be positive integer.')
        
    # The forward function defines the flow of the input data and thus decides which layer/chunk goes on top of what.
    def forward(self, src, nodes, edges, connections, edge_mask):
        out = self.MIF_ST(src, nodes, edges, connections, edge_mask)
        #print(out.shape)
        out=self.globalpool(out)
        #print(out.shape)
        out=self.fc(out)
        #print(out.shape)
        out=self.last_layer(out)
        #print(out.shape)
        return out