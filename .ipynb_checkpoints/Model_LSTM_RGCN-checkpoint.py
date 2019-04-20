from Model import *
import torch as th,torch
import numpy as np
from dgl.contrib.data import load_data
def domodel190420(dataset = 'mutag'):
    data = load_data(dataset=dataset)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_classes = data.num_classes
    labels = data.labels
    train_idx = data.train_idx
    # split training and validation set
    val_idx = train_idx[:len(train_idx) // 5]
    train_idx = train_idx[len(train_idx) // 5:]

    # edge type and normalization factor
    edge_type = torch.from_numpy(data.edge_type)
    edge_norm = torch.from_numpy(data.edge_norm).unsqueeze(1)

    labels = torch.from_numpy(labels).view(-1)

    # create graph
    g = DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(data.edge_src, data.edge_dst)
    g.edata.update({'type': edge_type.long(), 'norm': edge_norm})
    inputs = torch.arange(num_nodes).reshape(-1,1)

    RNN_Hidden_Size = []
    RGCN_Input_Size = []
    RGCN_Hidden_Size = []
    DROUPOUT = []
    Num_Bases = []
    Val_Acc = []
    Numbase = [40]
    if dataset == 'mutag':
        Numbase = [0,30]
    elif dataset == 'aifb':
        Numbase = [0]
    for RNN_hidden_size in [20,30,40,50]:
        for RGCN_input_size in [10,20,30,40]:
            for RGCN_hidden_size in [10,20,30,40]:
                for dropout in [0,0.1,0.2,0.3,0.4,0.5]:
                    for Num_bases in Numbase:
                        RNN_Hidden_Size.append(RNN_hidden_size)
                        RGCN_Input_Size.append(RGCN_input_size)
                        RGCN_Hidden_Size.append(RGCN_hidden_size)
                        RNN_input_size = num_nodes
                        DROUPOUT.append(dropout)
                        Num_Bases.append(Num_bases)
                        # RNN_hidden_size = 50
                        # RGCN_input_size = 40
                        # RGCN_hidden_size = 20
                        Num_classes = num_classes
                        Num_rels = num_rels
                        #dropout = 0.5
                        activation = F.relu
                        sequence_length = 1
                        #Num_bases=30
                        lr = 0.01 # learning rate
                        l2norm = 5e-4 # L2 norm coefficient
                        n_epochs = 50 # epochs to train

                        model = Model(RNN_input_size,
                                             RNN_hidden_size,
                                             RGCN_input_size,
                                             RGCN_hidden_size,
                                             Num_classes,
                                             Num_rels,
                                             Num_bases=Num_bases,
                                             Num_hidden_layers=0,
                                             dropout=dropout)

                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)
                        criterion = nn.CrossEntropyLoss()
                        print("start training...")
                        model.train()
                        for epoch in range(n_epochs):
                            optimizer.zero_grad()
                            logits = model.forward(g,inputs,sequence_length)
                            loss = criterion(logits[train_idx], labels[train_idx].long())
                            loss.backward()

                            optimizer.step()
                            train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx].long())
                            train_acc = train_acc.item() / len(train_idx)
                            if train_acc == 1:
                                break
                        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx].long())
                        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx].long())
                        val_acc = val_acc.item() / len(val_idx)
                        print("Epoch {:05d} | ".format(epoch) +
                                "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
                                    train_acc, loss.item()) +
                                "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
                                    val_acc, val_loss.item()))
                        Val_Acc.append(val_acc)
    c=[]
    for i in range(len(RNN_Hidden_Size)):
        c.append([RNN_Hidden_Size[i],RGCN_Input_Size[i],RGCN_Hidden_Size[i],DROUPOUT[i],Num_Bases[i],Val_Acc[i]])
    with open('result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in c:
            writer.writerow(row)
            
datasets = ['aifb','mutag','bgs','am']
for dataset in datasets:
    domodel190420(dataset)