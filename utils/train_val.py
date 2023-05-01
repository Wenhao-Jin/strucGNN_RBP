## With early stopping
from sklearn.metrics import roc_auc_score

def train(net, trainloader, optimizer, criterion, EGNN=False, node_feature_int=False, edge_3Di=False, has_edge_attr=True):
    net.train()
    running_loss = 0.0
    running_loss_total=0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        data=data.to(device)
        #(_, x), (_, edge_index), (_, edge_attr), (_, labels), (_, pos), (_, prot_name), (_, batch), (_, ptr) = data
        #inputs=inputs.double()
        #edge_index=torch.stack((edge_index[:,0], edge_index[:,1]),dim=0)
        if node_feature_int:
                data.x=data.x.long()
        else:
                data.x=data.x.double()
        if has_edge_attr:
            if edge_3Di:
                #edge_attr=edge_attr
                data.edge_attr=torch.nan_to_num(data.edge_attr, nan=0.0)
            else:
                data.edge_attr=data.edge_attr[:,None]
        else:
            data.edge_attr=None
        #edge_attr=None
        data.y=data.y[:,None]
        data.y=data.y.double()
        #print(prot_name)

        # zero the parameter gradients
        #optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(data.x, data.pos, data.batch)
        #loss = criterion(outputs, labels.double())
        #print('outputs:',outputs.shape)
        #print('labels:',labels.shape)
        loss = criterion(outputs, data.y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_loss_total += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            running_loss = 0.0            
    return net, running_loss_total / len(trainloader)

def val(net, loader, criterion, OnTestSet=False, EGNN=False, node_feature_int=False, edge_3Di=False, has_edge_attr=True):
    net.eval()
    correct=0
    y_test_pred_total=[]
    y_test_total=[]
    prot_names=[]
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            data=data.to(device)
            #(_, x), (_, edge_index), (_, edge_attr), (_, labels), (_, pos), (_, prot_name), (_, batch), (_, ptr) = data
            #inputs=inputs.double()
            #edge_index=torch.stack((edge_index[:,0], edge_index[:,1]),dim=0)
            if node_feature_int:
                data.x=data.x.long()
            else:
                data.x=data.x.double()
            if has_edge_attr:
                if edge_3Di:
                    data.edge_attr=torch.nan_to_num(data.edge_attr, nan=0.0)
                else:
                    data.edge_attr=data.edge_attr[:,None]
            else:
                data.edge_attr=None
            
            data.y=data.y[:,None]
            data.y=data.y.double()
            outputs = net(data.x, data.pos, data.batch)

            loss = criterion(outputs, data.y)
            running_loss += loss.item()  
            y_test_pred_total=np.concatenate((y_test_pred_total,outputs.numpy().flatten()))
            y_test_total=np.concatenate((y_test_total,data.y.numpy().flatten()))
            prot_names+=data.prot_name

    if OnTestSet:
        return running_loss /len(loader), roc_auc_score(y_test_total, y_test_pred_total), prot_names, y_test_pred_total, y_test_total        
#         for name, param in net.named_parameters():
#             print(name,param)
    else:
        return running_loss /len(loader), roc_auc_score(y_test_total, y_test_pred_total)

    

def train_MIF(net, trainloader, optimizer, criterion):
#     if collated==True and RBP_set==None:
#         raise ValueError("If 'collated=True', 'RBP_set' must be provided.")
    net.train()
    running_loss = 0.0
    running_loss_total=0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        src, nodes, edges, connections, edge_mask, labels, prot_id = data
        src=src.to(device).long()
        nodes=nodes.to(device).double()
        edges=edges.to(device).double()
        connections=connections.to(device).long()
        edge_mask=edge_mask.to(device).long()
        labels=labels.to(device)
            
        #labels=labels[:,None]
        labels=labels.double()
        #print(prot_name)

        # zero the parameter gradients
        #optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad()

        # forward + backward + optimize
        #print(edge_index.shape)
        outputs = net(src, nodes, edges, connections, edge_mask)
        #loss = criterion(outputs, labels.double())
        #print('outputs:',outputs.shape)
        #print('labels:',labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_loss_total += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            running_loss = 0.0   
            
    
    return net, running_loss_total / len(trainloader)

def val_MIF(net, loader, criterion, OnTestSet=False):
    net.eval()
    correct=0
    y_test_pred_total=[]
    y_test_total=[]
    prot_names=[]
    prot_ids=[]
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            #data=data.to(device)
            src, nodes, edges, connections, edge_mask, labels, prot_id = data
            src=src.to(device).long()
            nodes=nodes.to(device).double()
            edges=edges.to(device).double()
            connections=connections.to(device).long()
            edge_mask=edge_mask.to(device).long()
            labels=labels.to(device)
#             for e in [src, nodes, edges, connections, edge_mask, labels]:
#                 e.to(device)
            #inputs=inputs.double()
            #edge_index=torch.stack((edge_index[:,0], edge_index[:,1]),dim=0)

            #labels=labels[:,None]
            labels=labels.double()
            outputs = net(src, nodes, edges, connections, edge_mask)

            loss = criterion(outputs, labels)
            running_loss += loss.item()  
            y_test_pred_total=np.concatenate((y_test_pred_total,outputs.numpy().flatten()))
            y_test_total=np.concatenate((y_test_total,labels.numpy().flatten()))
            prot_ids+=prot_id

    if OnTestSet:
        return running_loss /len(loader), roc_auc_score(y_test_total, y_test_pred_total), prot_ids, y_test_pred_total, y_test_total        
#         for name, param in net.named_parameters():
#             print(name,param)
    else:
        return running_loss /len(loader), roc_auc_score(y_test_total, y_test_pred_total)
