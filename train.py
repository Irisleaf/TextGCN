import gc
import warnings
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
torch.cuda.empty_cache()

from layer import GCN
from utils import accuracy
from utils import macro_f1
from utils import preprocess_adj
from utils import show_graph_detail

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")
from config import set_args
args = set_args()


class PrepareData:
    def __init__(self):
        print("prepare data")

        graph = nx.read_weighted_edgelist(f"{args.graph_dir}/sentiment_data.txt"
                                          , nodetype=int)
        show_graph_detail(graph)
        adj = nx.to_scipy_sparse_matrix(graph,
                                        nodelist=list(range(graph.number_of_nodes())),
                                        # nodelist=list(range(64)),
                                        weight='weight',
                                        dtype=np.float)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = preprocess_adj(adj, is_sparse=True)

        self.nfeat_dim = int(graph.number_of_nodes())
        # print(self.nfeat_dim)
        # self.nfeat_dim = 64
        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))
        value = [1.] * self.nfeat_dim
        shape = (self.nfeat_dim, self.nfeat_dim)
        indices = torch.from_numpy(
                np.vstack((row, col)).astype(np.int64))
        values = torch.FloatTensor(value)
        shape = torch.Size(shape)

        self.features = torch.sparse.FloatTensor(indices, values, shape)

        data = pd.read_csv('data/sentiment_data.csv',names=['id','title','lable']).dropna()
        data.sample(frac=1).reset_index(drop=True)
        labels = data['lable'].values.tolist()
        label2id = {label: indx for indx, label in enumerate(set(labels))}
        indexs = [int(label2id[label]) for label in data['lable'].values.tolist()]

        self.labels = [label2id[label] for label in labels]
        self.nclass = len(label2id)
        self.train_lst, self.test_lst = indexs[int(len(indexs)/10):], indexs[:int(len(indexs)/10)]
        print('len(len(self.train_lst)',len(self.train_lst),len(self.test_lst))




class TextGCNTrainer:
    def __init__(self, model, pre_data):
        self.model = model
        self.device = args.device
        self.epochs = args.epoch
        self.set_seed()
        self.predata = pre_data

    def set_seed(self):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    def run_process(self):
        self.prepare_data()
        self.model = self.model(nfeat=self.nfeat_dim,
                                nhid=args.hidden_size,
                                nclass=self.nclass,
                                dropout=args.dropout)

        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.tensornize()
        self.train()
        self.test()


    def prepare_data(self):
        self.adj = self.predata.adj
        self.nfeat_dim = self.predata.nfeat_dim
        self.features = self.predata.features
        self.labels = self.predata.labels
        self.nclass = self.predata.nclass

        self.train_lst, self.val_lst = train_test_split(self.predata.train_lst,
                                                        # test_size=args.val_ratio,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        random_state=args.seed)
        self.test_lst = self.predata.test_lst
        print('len(len(self.train_lst)', len(self.train_lst), len(self.val_lst),len(self.test_lst))

    def tensornize(self):
        self.model = self.model.to(self.device)
        self.adj = self.adj.to(self.device)
        self.features = self.features.to(self.device)
        self.labels = torch.LongTensor(self.labels).to(self.device)
        self.train_lst = torch.LongTensor(self.train_lst).to(self.device)
        self.val_lst = torch.LongTensor(self.val_lst).to(self.device)


    def train(self):

        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0

        # 遍历model.parameters()返回的全局参数列表
        for param in self.model.parameters():
            mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')
        best_loss = 9999
        for epoch in range(10):
            self.model.train()
            self.optimizer.zero_grad()
            # print('self.train_lst:', len(self.train_lst), set(np.array(self.train_lst.cpu())), len(self.test_lst), set(np.array(self.test_lst)))
            # print('shape:',self.features.shape, self.adj.shape,self.train_lst.shape)
            logits = self.model.forward(self.features, self.adj)
            # print('self.logits.labels', logits.shape)
            # print(logits[self.train_lst].cpu().detach().numpy())
            # print('self.logits[self.train_lst].labels', logits[self.train_lst].shape)
            # print('self.labels.shape', self.labels.shape, set(np.array(self.labels.cpu())))
            # print('self.labels[self.train_lst]', self.labels[self.train_lst].shape,
            #       set(np.array(self.labels[self.train_lst].cpu())))
            # print(self.labels[self.train_lst][0:100])

            loss = nn.CrossEntropyLoss()(logits[self.train_lst],
                                  self.labels[self.train_lst])
            # loss = nn.CrossEntropyLoss()(logits[self.labels],self.labels)
            # print('--------------',logits[self.labels].shape)

            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()

            val_desc = self.val(self.val_lst)
            print("epoch:{}, val loss: {:.6f}, val acc: {:.6f}, f1: {:.6f}, precision: {:.6f}, recall: {:.6f}".
                  format(epoch,val_desc['loss'], val_desc['acc'], val_desc['macro_f1'], val_desc['precision'], val_desc['recall']))
            if val_desc['loss']<best_loss:
                torch.save(self.model.state_dict(), args.saved_model_path+'gcn_model')
                best_loss = val_desc['loss']

    @torch.no_grad()
    def val(self, x):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward(self.features, self.adj)
            loss = nn.CrossEntropyLoss()(logits[x],
                                  self.labels[x])
            acc = accuracy(logits[x],
                           self.labels[x])
            f1, precision, recall = macro_f1(logits[x],
                                             self.labels[x],
                                             num_classes=self.nclass)

            desc = {
                f"loss": loss.item(),
                "acc"           : acc,
                "macro_f1"      : f1,
                "precision"     : precision,
                "recall"        : recall,
            }
        return desc

    @torch.no_grad()
    def test(self):
        self.test_lst = torch.LongTensor(self.test_lst).to(self.device)
        test_desc = self.val(self.test_lst)
        print('Performing Testing')
        print("test loss: {:.6f}, test acc: {:.6f}, f1: {:.6f}, precision: {:.6f}, recall: {:.6f}".
              format(test_desc['loss'], test_desc['acc'], test_desc['macro_f1'], test_desc['precision'], test_desc['recall']))


def main():
    model = GCN
    print("2")
    predata = PrepareData()
    TextGCN = TextGCNTrainer(model=model, pre_data=predata)
    TextGCN.run_process()
    model.test()


if __name__ == '__main__':
    # main()
    model = GCN
    print("1")
    predata = PrepareData()
    TextGCN = TextGCNTrainer(model=model, pre_data=predata)
    TextGCN.run_process()



