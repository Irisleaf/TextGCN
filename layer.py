import math
import torch as th

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):
        support = th.spmm(infeatn, self.weight)
        output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, 2048)
        self.gc11 = GraphConvolution(2048, nclass)
        self.dropout = th.nn.Dropout(0.5)

        # self.gc1 = GraphConvolution(nfeat, 2048)
        # self.gc2 = GraphConvolution(2048, 512)
        # self.gc3 = GraphConvolution(512, 64)
        # self.gc4 = GraphConvolution(64, nclass)
        # self.dropout = th.nn.Dropout(0.5)


    def forward(self, x, adj):

        x = self.gc1(x, adj)
        x = th.relu(x)
        x = self.dropout(x)

        x = self.gc11(x, adj)

        # x = self.gc1(x, adj)
        # x = th.relu(x)
        #
        #
        # x = self.gc2(x, adj)
        # x = th.relu(x)
        #
        #
        # x = self.gc3(x, adj)
        # x = th.relu(x)
        # x = self.dropout(x)
        #
        # x = self.gc4(x, adj)

        return x

