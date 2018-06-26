import data
import numpy as np
from random import shuffle, Random
from collections import defaultdict


class Node2Vec:

    def __init__(self, A, d = 128, r = 10, l = 80, k = 10, p = 0.01, q = 1):
        self.edges, self.vertices = [], []
        self.graph = {}
        self.adjacents = defaultdict(list)
        self.data_to_graph(A)
        self.preprocess_modified_weights(p, q)
        return

    def train(self):
        return

    def learn_features(self):
        return

    def node2vecWalk(self):
        return

    def alias_setup(self, prob, n):
        q = np.zeros(n)
        J = np.zeros(n, dtype=np.int)
        smaller = []
        larger = []
        for k in range(0, n):
            q[k] = n * prob
            if q[k] < 1.0:
                smaller.append(k)
            else:
                larger.append(k)

        while len(larger) > 0 and len(smaller) > 0 :
            small = smaller.pop()
            large = larger.pop()
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def preprocess_modified_weights(self, p, q):
        prob = 1.0 / len(self.vertices)
        nodes_p = {}
        for v in self.vertices:
            nodes_p[v] = self.alias_setup(prob, len(self.adjacents[v]))
        return

    def get_neighbors(self):
        return

    def alias_sample(self):
        return

    def data_to_graph(self, A):
        for i in range(A.shape[0]):
            self.vertices.append(i)
            for v in A[i].indices:
                if i in self.graph.keys():
                    self.graph[i][v] = {'weight': 1}
                else:
                    self.graph[i] = {v: {'weight': 1}}
                self.adjacents[i].append(v)
                self.edges.append((i, v))
        return
