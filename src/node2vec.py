import data
import numpy as np
from random import shuffle
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression


class Node2Vec:

    def __init__(self, A, d = 128, r = 10, l = 80, k = 10, p = 4, q = 0.25):
        self.edges, self.nodes = [], []
        self.graph = {}
        self.adjacents = defaultdict(list)
        self.data_to_graph(A)
        self.model = self.learn_features(d, r, l, k, p, q)
        return

    def train(self, y):
        print("Node2Vec training...")
        y = np.ravel(np.array([np.where(y[i] == 1)[0] for i in range(y.shape[0])]))
        x = np.array([self.model.wv[str(i)] for i in range(len(self.nodes))])

        y_train, y_val, y_test, idx_train, idx_val, idx_test = data.get_splits(y)
        x_train, x_val, x_test, idx_train, idx_val, idx_test = data.get_splits(x)

        regression = LogisticRegression()
        regression.fit(x_train, y_train)
        print("Result from validation data : ", regression.score(x_val, y_val))
        print("Result from test data : ", regression.score(x_test, y_test))
        return

    def learn_features(self, d, r, l, k, p, q):
        nodes_p, edges_p = self.preprocess_modified_weights(p, q)
        walks = []
        print("Node2Vec's preprocessing done...")
        for i in range(r):
             shuffle(self.nodes)
             for n in self.nodes:
                 walk = self.node2vecWalk(n, l, nodes_p, edges_p)
                 walks.append(walk)
        print("Node2VecWalks done...")
        walks = list([list(map(str, walk)) for walk in walks])
        model = Word2Vec(walks, size=d, window=k, min_count=0, sg=1, workers=4)
        return model

    def alias_setup(self, prob, n):
        s = []
        l = []
        q = np.zeros(n)
        J = np.zeros(n, dtype=np.int)
        for k in range(0, n):
            q[k] = n * prob
            if q[k] < 1.0:
                s.append(k)
            else:
                l.append(k)
        while len(l) > 0 and len(s) > 0:
            small = s.pop()
            large = l.pop()
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                s.append(large)
            else:
                l.append(large)
        return J, q

    def get_alias_edge(self, source, target, p, q):
        edge_weights, s, l = [], [], []
        for t in sorted(self.adjacents[target]):
            f = self.graph[target][t]['weight']
            if t == source:
                f = f / p
            elif f >= 1e-4:
                f = f
            else:
                f = f / q
            edge_weights.append(f)
        total = sum(edge_weights)
        normalized_edge_weights = [float(weight)/total for weight in edge_weights]
        n = len(normalized_edge_weights)
        q = np.zeros(n)
        J = np.zeros(n, dtype=np.int)
        for k, w in enumerate(normalized_edge_weights):
            q[k] = n * w
            if q[k] < 1.0:
                s.append(k)
            else:
                l.append(k)
        while len(l) > 0 and len(s) > 0:
            small = s.pop()
            large = l.pop()
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                s.append(large)
            else:
                l.append(large)
        return J, q

    def preprocess_modified_weights(self, p, q):
        nodes_p, edges_p = {}, {}
        M = len(self.nodes)
        N = len(self.edges)
        prob = 1.0 / len(self.nodes)
        for i in range(M):
            v = self.nodes[i]
            nodes_p[v] = self.alias_setup(prob, len(self.adjacents[v]))
        for i in range(N):
            e = self.edges[i]
            edges_p[e] = self.get_alias_edge(e[0], e[1], p, q)
        return nodes_p, edges_p


    def node2vecWalk(self, u, l, nodes_p, edges_p):
        walk = [u]
        while len(walk) < l:
            cur = walk[-1]
            v_curr = self.get_neighbors(cur)
            if len(v_curr) > 0:
                if len(walk) == 1:
                    a = nodes_p[cur][0]
                    b = nodes_p[cur][1]
                else:
                    a = edges_p[(walk[-2], cur)][0]
                    b = edges_p[(walk[-2], cur)][1]
                walk.append(v_curr[self.alias_sample(a, b)]
)
            else:
                break
        return walk

    def get_neighbors(self, cur):
        return sorted(self.adjacents[cur])

    def alias_sample(self, j, q):
        n = int(np.floor(np.random.rand() * len(j)))
        if np.random.rand() < q[n]:
            return n
        return j[n]

    def data_to_graph(self, A):
        for i in range(A.shape[0]):
            self.nodes.append(i)
            for v in A[i].indices:
                if i in self.graph.keys():
                    self.graph[i][v] = {'weight': 1}
                else:
                    self.graph[i] = {v: {'weight': 1}}
                self.adjacents[i].append(v)
                self.edges.append((i, v))
        return
