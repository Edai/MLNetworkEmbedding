import data
import numpy as np
from random import shuffle, Random
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression


class DeepWalk:
    def __init__(self, A, embedding_size=64, window_size=5, number_walks=10, walk_length=40):
        print("Deep Walk initialization...")
        self.edges, self.vertices = [], []
        self.graph = {}
        self.data_to_graph(A)
        walks = []
        for i in range(number_walks):
            shuffle(self.vertices)
            for v in self.vertices:
                walks.append(self.random_walk(walk_length, v, alpha=0))
        self.model = Word2Vec(walks, size=embedding_size, window=window_size, min_count=0, sg=1, hs=1, workers=4)
        return

    def train(self, y):
        print("Deep Walk training...")
        y = np.ravel(np.array([np.where(y[i] == 1)[0] for i in range(y.shape[0])]))
        x = np.array([self.model.wv[str(i)] for i in range(len(self.vertices))])

        y_train, y_val, y_test, idx_train, idx_val, idx_test = data.get_splits(y)
        x_train, x_val, x_test, idx_train, idx_val, idx_test = data.get_splits(x)

        regression = LogisticRegression(C=0.1)
        regression.fit(x_train, y_train)
        print("Result from validation data : ", regression.score(x_val, y_val))
        print("Result from test data : ", regression.score(x_test, y_test))
        return

    def data_to_graph(self, A):
        for i in range(A.shape[0]):
            self.vertices.append(i)
            for v in A[i].indices:
                self.edges.append((i, v))
        for v in self.vertices:
            self.graph[v] = []
            for e in self.edges:
                if e[0] == v:
                    self.graph[v].append(e[1])
        return

    def random_walk(self, walk_length, start, alpha=0):
        rand = Random()
        path = [start]
        while len(path) < walk_length:
            cur = path[len(path) - 1]
            v = self.graph[cur]
            if len(v) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(v))
                else:
                    path.append(path[0])
            else:
                return [str(node) for node in path]
        return [str(node) for node in path]
