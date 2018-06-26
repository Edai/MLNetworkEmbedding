import data
import deepwalk as dw
import node2vec as nv
import line as l
import aane as aa

from pprint import pprint

def deep_walk_cora():
    X, A, y = data.load_data(dataset='cora')
    deepwalk = dw.DeepWalk(A);
    return deepwalk.train(y)


def line_tencent():
    adj, labels = data.load_data(dataset='tencent', path='data/tencent/')
    return

def node2vec_cora():
    X, A, y = data.load_data(dataset='cora')
    node2vec = nv.Node2Vec(A)
    return node2vec.train()

def main():
    #deep_walk_cora()
    #line_tencent()
    node2vec_cora()
    return

if __name__ == '__main__':
    main()
