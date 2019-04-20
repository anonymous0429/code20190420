from conv import GraphConv
from RGCN.layers import RGCNBasisLayer
import torch.nn as nn
import dgl
import dgl.function as fn
import torch as th,torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from dgl import DGLGraph
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from functools import partial
#留给GPU的接口
def create_variable(tensor):
    return Variable(tensor)


class RNNEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1

        self.embedding = nn.Embedding(input_size, hidden_size)
        # self.gru = nn.GRU(hidden_size, hidden_size, n_layers,bidirectional=bidirectional)
        self.LSTM = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, seq_lengths):
        # Note: we run this all at once (over the whole input sequence)
        # input shape: B x S (input size)
        # transpose to make S(sequence) x B (batch)
        input = input.t()
        batch_size = input.size(1)

        # Make a hidden
        # hidden = self._init_hidden(batch_size)
        h_0 = torch.zeros(self.n_layers * self.n_directions,
                          batch_size, self.hidden_size)
        c_0 = torch.zeros(self.n_layers * self.n_directions,
                          batch_size, self.hidden_size)
        # Embedding S x B -> S x B x I (embedding size)
        # print("s*b",input.size())
        embedded = self.embedding(input.long())
        # print("s*b*i",embedded.size())
        # Pack them up nicely
        # gru_input = pack_padded_sequence(embedded, seq_lengths.data.cpu().numpy())

        # To compact weights again call flatten_parameters().

        # self.gru.flatten_parameters()
        # output, hidden = self.gru(embedded, hidden)

        self.LSTM.flatten_parameters()

        output, (h_n, c_n) = self.LSTM(embedded, (h_0, c_0))
        # Use the last layer output as FC's input
        # No need to unpack, since we are going to use hidden
        fc_output = self.fc(h_n[-1, :])
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size)
        return create_variable(hidden)


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

        # create initial features
        #self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, features):
        self.features = features
        if self.features is not None:
            g.ndata['h'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')


class RGCN(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.num_nodes)
        if self.use_cuda:
            features = features.cuda()
        return features

    def build_input_layer(self):
        return RGCNBasisLayer(self.num_nodes, self.h_dim, self.num_rels, self.num_bases, activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self, idx):
        return RGCNBasisLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNBasisLayer(self.h_dim, self.out_dim, self.num_rels,self.num_bases)


class Model(nn.Module):
    def __init__(self, RNN_input_size, RNN_hidden_size, RGCN_input_size, RGCN_hidden_size, Num_classes, Num_rels,
                 Num_bases=-1, Num_hidden_layers=1, dropout=0):
        super(Model, self).__init__()

        self.RNN = RNNEncoder(RNN_input_size, RNN_hidden_size, RGCN_input_size)

        self.RGCN = RGCN(RGCN_input_size,
                         RGCN_hidden_size,
                         Num_classes,
                         Num_rels,
                         Num_bases,
                         Num_hidden_layers,
                         dropout)

    def forward(self, g, inputs, sequence_length):
        features = self.RNN(inputs, sequence_length)  # RNN编码

        x = self.RGCN(g, features)
        # x = self.gcn1(g, features)
        return x