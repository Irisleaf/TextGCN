import gc
import warnings
from time import time
import random
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import manifold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader,TensorDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from layer import GCN
from utils import accuracy
from utils import macro_f1
from utils import preprocess_adj
from utils import show_graph_detail
from sklearn.metrics import confusion_matrix, classification_report
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")
from config import set_args
args = set_args()
device = args.device

def Prepare_Sentiment_txt(path):
    print("deal with sentiment_txt")

    graph = nx.read_weighted_edgelist(path,nodetype=int)
    show_graph_detail(graph)
    adj = nx.to_scipy_sparse_matrix(graph,
                                    nodelist=list(range(graph.number_of_nodes())),
                                    weight='weight',
                                    dtype=np.float)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = preprocess_adj(adj, is_sparse=True)

    nfeat_dim = int(graph.number_of_nodes())
    row = list(range(nfeat_dim))
    col = list(range(nfeat_dim))
    value = [1.] * nfeat_dim
    shape = (nfeat_dim, nfeat_dim)
    indices = torch.from_numpy(
            np.vstack((row, col)).astype(np.int64))
    values = torch.FloatTensor(value)
    shape = torch.Size(shape)

    features = torch.sparse.FloatTensor(indices, values, shape)

    return adj.to(args.device),features.to(args.device),nfeat_dim

def Prepare_Sentiment_csv(path):
    print("deal with sentiment_csv")
    data = pd.read_csv(path,names=['id','title','lable']).dropna()
    data.sample(frac=1).reset_index(drop=True)

    labels = data['lable'].values.tolist()
    label2id = {label: indx for indx, label in enumerate(set(labels))}
    indexs = [int(label2id[label]) for label in data['lable'].values.tolist()]
    indexs = [i for i in range(len(indexs))]
    labels = [label2id[label] for label in labels]
    nclass = len(label2id)

    return nclass,labels,indexs,label2id

def get_data(indexs,labels):
    lens = len(indexs)
    index_ = indexs.copy()
    random.shuffle(index_)
    indexs = [indexs[i] for i in index_]
    labels = [labels[i] for i in index_]
    train_lst, val_lst, test_lst = indexs[:int(lens / 10) * 7], indexs[int(lens / 10) * 7:int(lens / 10) * 8], indexs[int(lens / 10) * 8:]
    train_labels,val_labels,test_labels = labels[:int(lens / 10) * 7],labels[int(lens / 10) * 7:int(lens / 10) * 8],labels[int(lens / 10) * 8:]
    train_labels = torch.LongTensor(train_labels).to(device)
    val_labels = torch.LongTensor(val_labels).to(device)
    test_labels = torch.LongTensor(test_labels).to(device)
    train_lst = torch.LongTensor(train_lst).to(device)
    val_lst = torch.LongTensor(val_lst).to(device)
    test_lst = torch.LongTensor(test_lst).to(device)
    return train_lst, val_lst, test_lst,train_labels,val_labels,test_labels

def train(model,optimizer,indexs,labels,features, adj,nclass,label2id,epochs=10,best_loss=0.7,lr=0.01,ratio=0.9):
    model.train()
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    for epoch in range(epochs):
        train_lst, val_lst, test_lst, train_labels, val_labels, test_labels = get_data(indexs, labels)
        data_set = TensorDataset(train_lst, train_labels)
        data_loader = DataLoader(data_set, batch_size=len(train_lst))
        for train_lst, train_labels in data_loader:
            optimizer.zero_grad()
            logits = model.forward(features, adj)
            loss = nn.CrossEntropyLoss()(logits[train_lst], train_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        val_desc = val(model,val_lst,val_labels,features, adj,nclass)
        print("epoch:{}, val loss: {:.6f}, val acc: {:.6f}, f1: {:.6f}, precision: {:.6f}, recall: {:.6f}".
              format(epoch, val_desc['loss'], val_desc['acc'], val_desc['macro_f1'], val_desc['precision'],
                     val_desc['recall']))
        if val_desc['loss'] < best_loss or epoch==epochs-1:
            savepath = args.saved_model_path + 'gcn_model_'+str(nclass)+'.pth'
            torch.save(model.state_dict(), savepath)
            best_loss = val_desc['loss']
    test_desc = val(model,test_lst, test_labels,features, adj,nclass,type='test',label2id=label2id)
    print('Performing Testing')
    print("test loss: {:.6f}, test acc: {:.6f}, f1: {:.6f}, precision: {:.6f}, recall: {:.6f}".
          format(test_desc['loss'], test_desc['acc'], test_desc['macro_f1'], test_desc['precision'],
                 test_desc['recall']))



def val(model,val_lst,val_labels,features, adj,nclass,type='val',label2id=None):
    model.eval()
    with torch.no_grad():
        logits = model.forward(features, adj)
        loss = nn.CrossEntropyLoss()(logits[val_lst],
                              val_labels)
        acc = accuracy(logits[val_lst],
                    val_labels)
        f1, precision, recall = macro_f1(logits[val_lst],
                                         val_labels,
                                         num_classes=nclass)
        if type == 'test':
            plot_confusion_matrix(np.array(logits[val_lst].argmax(dim=1)),np.array(val_labels), label2id)
            plot_tsne(np.array(logits[val_lst].argmax(dim=1)),np.array(val_labels), label2id)
        desc = {
            f"loss": loss.item(),
            "acc"           : acc,
            "macro_f1"      : f1,
            "precision"     : precision,
            "recall"        : recall,
        }
    return desc

def plot_confusion_matrix(predicted_output, true_output, labels_dict, description='Description', normalize=True):

    # Plots a confusion matrix for the predicted output.

    # Parameters:
    #     description: {str} Description.
    #     predicted_output: {numpy.array} Predicted output.
    #     true_output: {numpy.array} True output.
    #     labels_dict: {dict} Dictionary of class labels.
    #     normalize: {bool} Normalize values or not.

    title = "Confusion matrix: suicide '%s'" % (max(true_output)+1)
    if normalize:
        title = title + " (Normalized)"
    cm = confusion_matrix(true_output, predicted_output)
    classes = get_classes_stats(predicted_output, true_output)[0]
    labels = []
    dict_new = {value:key for key,value in labels_dict.items()}
    for item in classes:
        labels.append(dict_new[item])
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    cmap = plt.cm.Purples
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]), yticklabels=labels)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_title(title, fontsize=20)
    ax.set_ylabel("True labels", fontsize=20)
    ax.set_xlabel("Predicted labels", fontsize=20)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    ax.grid(False)
    savepath = './savepig/res'+ str(max(true_output)+1) +'.png'
    plt.savefig(savepath)

def plot_tsne(x, y, true_output):

    tsne = manifold.TSNE(n_components=2,init='pca', random_state=501)
    # x = x.reshape(1, -1)
    x_tsne = tsne.fit_transform(x)
    print("Orgi data dimension is {}. Embedded data dimension is {}".format(x.shape[-1], x_tsne.shape[-1]))

    x_min, x_max = x_tsne.min(0), x_tsne.max(0)
    x_norm = (x_tsne - x_min) / (x_max - x_min)  # normalization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    for i in range(x_norm.shape[0]):
        ax.text(x_norm[i, 0], x_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    ax.set_title('t-nse', fontweight="bold")
    ax.grid(True)

    plt.show()
    savepath = './savepig/tnse'+ str(max(true_output)+1) +'.png'
    plt.savefig(savepath)

def get_classes_stats(predicted_output, true_output):
    # Returns the calculated evaluation metrics for each class.

    # Parameters:
    #     predicted_output: {numpy.array} Predicted output.
    #     true_output: {numpy.array} True output.

    report_dict = classification_report(true_output, predicted_output, output_dict=True)
    classes = []
    classes_stats = []
    for key in report_dict:
        if key not in ["accuracy", "macro avg", "weighted avg"]:
            classes.append(int(float(key)))
            stats = [float("%0.3f" % (report_dict[key]['precision'])), float("%0.3f" % (report_dict[key]['recall'])),
                     float("%0.3f" % (report_dict[key]['f1-score'])), report_dict[key]['support']]
            classes_stats.append(stats)
    return [classes, classes_stats]

