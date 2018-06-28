import data
import deepwalk as dw
import node2vec as nv
import line
import aane as aa
import numpy as np
import sklearn.metrics as sk
from functools import reduce
import argparse


def deep_walk_cora():
    print("DEEP WALK")
    X, A, y = data.load_data(dataset='cora')
    deepwalk = dw.DeepWalk(A);
    return deepwalk.train(y)


def line_tencent():
    print("LINE")
    adj = data.load_data(dataset='tencent', path='data/tencent/', file_tencent='adj_train.npz')
    a = line.Line(adj)
    e1 = a.train(proximity="first_order")
    e2 = a.train(proximity="second_order")
    embedding = np.concatenate((e1, e2), axis=1)
    val_positives_edges = np.load('data/tencent/val_edges.npy')
    val_false_edges = np.load('data/tencent/val_edges_false.npy')
    test_positive_edges = np.load('data/tencent/test_edges.npy')
    test_negative_edges = np.load('data/tencent/test_edges_false.npy')
    positive_val, negative_val, positive_cosine, negative_cosine = [], [], [], []
    y_test = [1] * len(test_positive_edges) + [0] * len(test_negative_edges)
    for edge in val_positives_edges:
        positive_val.append(sk.pairwise.cosine_similarity(embedding[edge[0], :].reshape((1, -1)),
                                                          embedding[edge[1], :].reshape((1, -1))))
    for edge in val_false_edges:
        negative_val.append(sk.pairwise.cosine_similarity(embedding[edge[0], :].reshape((1, -1)),
                                                          embedding[edge[1], :].reshape((1, -1))))
    for edge in test_positive_edges:
        positive_cosine.append(sk.pairwise.cosine_similarity(embedding[edge[0], :].reshape((1, -1)),
                                                             embedding[edge[1], :].reshape((1, -1))))
    for edge in test_negative_edges:
        negative_cosine.append(sk.pairwise.cosine_similarity(embedding[edge[0], :].reshape((1, -1)),
                                                             embedding[edge[1], :].reshape((1, -1))))
    t = (reduce((lambda x, y: x + y), positive_val) /
         len(positive_val) + reduce((lambda x, y: x + y), negative_val) / len(negative_val)) / 2
    positive_cosine = [1 if a > t else 0 for a in positive_cosine]
    negative_cosine = [1 if a > t else 0 for a in negative_cosine]
    y_score = positive_cosine + negative_cosine
    print(sk.roc_auc_score(y_test, y_score))
    return


def aane_tencent():
    print("AANE")
    d = data.load_data(dataset='tencent', path='data/tencent/', file_tencent='adj_train.npz')
    a = aa.AANE(d)
    return a.train()


def node2vec_cora():
    print("NODE2VEC")
    X, A, y = data.load_data(dataset='cora')
    node2vec = nv.Node2Vec(A)
    return node2vec.train(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deepwalk', default=0, required=False)
    parser.add_argument('--line', default=0, required=False)
    parser.add_argument('--node2vec', default=0, required=False)
    parser.add_argument('--aane', default=0, required=False)

    args = parser.parse_args()
    if args.deepwalk is not 0:
        deep_walk_cora()
    if args.line is not 0:
        line_tencent()
    if args.node2vec is not 0:
        node2vec_cora()
    if args.aane is not 0:
        aane_tencent()
    return


if __name__ == '__main__':
    main()
