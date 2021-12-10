import numpy as np
import torch as th
import scipy.sparse as sp
import networkx as nx
from sklearn import metrics

def macro_f1(pred, targ, num_classes=None):
    pred = th.max(pred, 1)[1]
    # print('set:',set(np.array(pred.cpu())))
    f1_ = metrics.f1_score(list(np.array(targ.cpu())), list(np.array(pred.cpu())), labels=[i for i in range(num_classes)],average='macro')
    precision_ = metrics.precision_score(list(np.array(targ.cpu())), list(np.array(pred.cpu())),
                                        labels=[i for i in range(num_classes)],average='macro')
    recall_ = metrics.recall_score(list(np.array(targ.cpu())), list(np.array(pred.cpu())),
                                  labels=[i for i in range(num_classes)],average='macro')
    # print('----',(pred != targ).sum())
    # C2 = metrics.confusion_matrix(list(np.array(targ.cpu())), list(np.array(pred.cpu())), labels=[i for i in range(num_classes)])
    # print(C2)
    tp_out = []
    fp_out = []
    fn_out = []
    # print(num_classes)
    if num_classes is None:
        num_classes = sorted(set(targ.cpu().numpy().tolist()))
    else:
        num_classes = range(num_classes)
    for i in num_classes:
        tp = ((pred == i) & (targ == i)).sum().item()  # 预测为i，且标签的确为i的/被判定为正样本，事实上也是证样本
        fp = ((pred == i) & (targ != i)).sum().item()  # 预测为i，但标签不是为i的/被判定为正样本，但事实上是负样本。
        fn = ((pred != i) & (targ == i)).sum().item()  # 预测不是i，但标签是i的/被判定为负样本，但事实上是正样本
        # print(tp,fp,fn)
        tp_out.append(tp)
        fp_out.append(fp)
        fn_out.append(fn)

    eval_tp = np.array(tp_out)
    eval_fp = np.array(fp_out)
    eval_fn = np.array(fn_out)
    # print('eval_tp:',eval_tp)
    # print('eval_fp:',eval_fp)
    # print('eval_fn:',eval_fn)

    precision = eval_tp / (eval_tp + eval_fp)
    precision[np.isnan(precision)] = 0
    precision = np.mean(precision)

    recall = eval_tp / (eval_tp + eval_fn)
    recall[np.isnan(recall)] = 0
    recall = np.mean(recall)

    f1 = 2 * (precision * recall) / (precision + recall)
    # return f1_, precision_, recall_

    return f1, precision, recall

def accuracy(pred, targ):
    pred = th.max(pred, 1)[1]
    acc = ((pred == targ).float()).sum().item() / targ.size()[0]

    return acc



def show_graph_detail(graph):

    dst = {"nodes"    : nx.number_of_nodes(graph),
           "edges"    : nx.number_of_edges(graph),
           "selfloops": nx.number_of_selfloops(graph),
           "isolates" : nx.number_of_isolates(graph),
           "覆盖度"      : 1 - nx.number_of_isolates(graph) / nx.number_of_nodes(graph), }
    show_table(dst)


def show_table(dst):
    table_title = list(dst.keys())
    from prettytable import PrettyTable
    table = PrettyTable(field_names=table_title, header_style="title", header=True, border=True,
                        hrules=1, padding_width=2, align="c")
    table.float_format = "0.4"
    table.add_row([dst[i] for i in table_title])
    # print('table:',table)


def preprocess_adj(adj, is_sparse=False):
    """Preprocessing of adjacency matrix for simple pygGCN model and conversion to
    tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    else:
        return th.from_numpy(adj_normalized.A).float()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


