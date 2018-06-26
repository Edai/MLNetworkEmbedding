import data
import numpy as np
from random import shuffle, Random


class Node2Vec:

    def __init__(self, A):
        self.edges, self.vertices = [], []
        self.graph = {}
        self.data_to_graph(A)
        return

    def train(self):
        return

    def data_to_graph(self, A):
        for i in range(A.shape[0]):
            self.vertices.append(i)
            for v in A[i].indices:
                self.edges.append((i, v))
                if i in self.graph.keys():
                    self.graph[i][v] = {'weight': 1}
                else:
                    self.graph[i] = {v: {'weight': 1}}
        return