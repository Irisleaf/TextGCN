import os
from collections import Counter

import networkx as nx

import itertools
import math
import re
from config import set_args
from collections import defaultdict
from time import time
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from nltk.corpus import stopwords
from utils import show_graph_detail
args = set_args()


def get_window(content_lst, window_size):

    word_window_freq = defaultdict(int)  # w(i)  单词在窗口单位内出现的次数
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_len = 0
    for words in tqdm(content_lst, desc="Split by window"):
        windows = list()
        if isinstance(words, str):
            words = words.split()
        length = len(words)

        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(list(set(window)))

        for window in windows:
            for word in window:
                word_window_freq[word] += 1

            for word_pair in itertools.combinations(window, 2):
                word_pair_count[word_pair] += 1

        windows_len += len(windows)
    return word_window_freq, word_pair_count, windows_len


def cal_pmi(W_ij, W, word_freq_1, word_freq_2):
    p_i = word_freq_1 / W
    p_j = word_freq_2 / W
    p_i_j = W_ij / W
    pmi = math.log(p_i_j / (p_i * p_j))

    return pmi


def count_pmi(windows_len, word_pair_count, word_window_freq, threshold):
    word_pmi_lst = list()
    for word_pair, W_i_j in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        word_freq_1 = word_window_freq[word_pair[0]]
        word_freq_2 = word_window_freq[word_pair[1]]

        pmi = cal_pmi(W_i_j, windows_len, word_freq_1, word_freq_2)
        if pmi <= threshold:
            continue
        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])
    return word_pmi_lst


def get_pmi_edge(content_lst, window_size=20, threshold=0.):
    word_window_freq, word_pair_count, windows_len = get_window(content_lst,
                                                                window_size=window_size)

    pmi_edge_lst = count_pmi(windows_len, word_pair_count, word_window_freq, threshold)
    return pmi_edge_lst


class BuildGraph:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.graph_dir = args.graph_dir

        self.word2id = dict()
        self.g = nx.Graph()
        self.content = self.read_file(self.file_dir)

        self.get_tfidf_edge()
        self.get_pmi_edge()
        self.save()

    def read_file(self, file_dir):
        data = pd.read_csv(file_dir).dropna()
        sent = data['title'].values.tolist()
        labels = data['lable'].values.tolist()
        sent = [self.clean_text(line) for line in sent]
        assert len(sent) == len(labels),'match error'
        return sent

    def clean_text(self, text):
        text = text.strip('\n').lower()
        text = text.replace(',','').replace('.','').replace('?','').replace('...','').replace('!','').replace('!','').replace('<num>','').replace('~','')

        # stop_words = set(stopwords.words('english'))
        # text_new = list()
        # for word in text:
        #     if word in stop_words:
        #         continue
        #     text_new.append(word)
        # return " ".join(text_new)
        return text

    def get_pmi_edge(self):

        pmi_edge_lst = get_pmi_edge(self.content, window_size=4, threshold=0.0)

        for edge_item in pmi_edge_lst:

            word_indx1 = self.node_num + self.word2id[edge_item[0]]
            word_indx2 = self.node_num + self.word2id[edge_item[1]]
            if word_indx1 == word_indx2:
                continue
            self.g.add_edge(word_indx1, word_indx2, weight=edge_item[2])

        show_graph_detail(self.g)

    def get_tfidf_edge(self):
        # 获得tfidf权重矩阵（sparse）和单词列表
        tfidf_vec = self.get_tfidf_vec()

        count_lst = list()  # 统计每个句子的长度
        for ind, row in tqdm(enumerate(tfidf_vec),
                             desc="generate tfidf edge"):
            count = 0
            for col_ind, value in zip(row.indices, row.data):
                word_ind = self.node_num + col_ind
                self.g.add_edge(ind, word_ind, weight=value)
                count += 1
            count_lst.append(count)

        show_graph_detail(self.g)

    def get_tfidf_vec(self):
        """
        学习获得tfidf矩阵，及其对应的单词序列
        :param content_lst:
        :return:
        """
        start = time()
        text_tfidf = Pipeline([
            ("vect", CountVectorizer(min_df=1,
                                     max_df=1.0,
                                     token_pattern=r"\S+",
                                     )),
            ("tfidf", TfidfTransformer(norm=None,
                                       use_idf=True,
                                       smooth_idf=False,
                                       sublinear_tf=False
                                       ))
        ])

        tfidf_vec = text_tfidf.fit_transform(self.content)

        self.node_num = tfidf_vec.shape[0]

        vocab_lst = text_tfidf["vect"].get_feature_names()
        print(vocab_lst)

        print("vocab_lst len:", len(vocab_lst))
        for ind, word in enumerate(vocab_lst):
            self.word2id[word] = ind

        self.vocab_lst = vocab_lst

        return tfidf_vec

    def save(self):
        nx.write_weighted_edgelist(self.g,
                                   f"{self.graph_dir}/sentiment_data.txt")




BuildGraph("data/sentiment_data.csv")


