import ProcessData
from dgl import DGLGraph
import dgl
import dgl.function as fn
import torch
import numpy as np


def LoadData(filename):

    xml = filename + ".xml"
    trace="Data\\"+filename+"\\"
    title,labels,jconf,authors,FullName,organization = ProcessData.ProcessingRawData(trace+xml)
    title = ProcessData.Wipe_off_Punctuation(title)
    title_vocab,title_split = ProcessData.Split_Title(title)
    title_one_hot,Max_Sequence_Len,vocab_size = ProcessData.One_hot_encoding(title_vocab,title_split)
    title_one_hot_padding = ProcessData.Padding_One_hot(title_one_hot,Max_Sequence_Len)
    author_vocab,authors_split = ProcessData.Split_Authors(authors)
    
    edge_type = np.load(trace+"edge_type.npy")
    edge_list_src = np.load(trace+"edge_list_src.npy")
    edge_list_dst = np.load(trace+"edge_list_dst.npy")
    
    num_nodes = len(authors_split)
    edge_norm = [1 for i in range(len(edge_type))]
    

    print("Number of edges: ",len(edge_list_src))
    print("Number of nodes: ",len(authors_split))
    print("Number of class: ",max(labels)+1)
    
    train_idx = np.load(trace+"train_index.npy")
    test_idx = np.load(trace+"test_index.npy")
    
    inputs = title_one_hot_padding
    labels = labels
    return edge_type,edge_list_src,edge_list_dst,num_nodes,edge_norm,vocab_size,train_idx,test_idx,inputs,labels