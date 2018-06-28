import networkx as nx
import numpy as np
import tensorflow as tf


class Line:

    def __init__(self, data):
        self.g = nx.from_scipy_sparse_matrix(data)
        self.degree = list(self.g.degree)
        self.nodes = list(self.g.nodes)
        self.edges = list(self.g.edges)
        self.node_distribution = np.power(
            np.array([self.degree[node][1] for node in self.nodes], dtype=np.float32), 0.75)
        self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.g.edges(data=True)])
        self.node_distribution /= np.sum(self.node_distribution)
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.node_sampling = AliasSampling(prob=self.node_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        return

    def fetch_batch(self, batch_size=16, K=10):
        edge_batch_index = self.edge_sampling.sampling(batch_size)
        u_i = []
        u_j = []
        label = []
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            for i in range(K):
                while True:
                    negative_node = self.node_sampling.sampling()
                    if self.g.has_edge(negative_node, edge[1]) == False:
                        break
                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
        return u_i, u_j, label

    def train(self, batch_size=16, dimension=128, num_batches=1000, K=10, proximity='second_order',
              learning_rate=0.025):
        model = LineTF(num_of_nodes=len(self.nodes), dimension=dimension, batch_size=batch_size,
                       K=K, proximity=proximity, learning_rate=learning_rate)
        sess = model.sess;
        sess.run(tf.global_variables_initializer())
        sess.run(model.embedding)
        for i in range(0, num_batches):
            u_i, u_j, label = self.fetch_batch(batch_size=batch_size, K=K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            if i % 100 is not 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                if learning_rate > learning_rate * 1e-4:
                    learning_rate = learning_rate * (1 - i / num_batches)
                else:
                    learning_rate = learning_rate * 1e-4
            else:
                sess.run(model.loss, feed_dict=feed_dict)
        embedding = sess.run(model.embedding)
        normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        print(type(normalized_embedding))
        return normalized_embedding


class LineTF:
    def __init__(self, K, proximity, batch_size, dimension, num_of_nodes, learning_rate):
        self.sess = tf.Session()
        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[batch_size * (K + 1)])
        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[batch_size * (K + 1)])
        self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[batch_size * (K + 1)])
        with tf.variable_scope(proximity, reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable('target_embedding', [num_of_nodes, dimension],
                                             initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
        self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=num_of_nodes), self.embedding)
        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
        if proximity == 'first_order':
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=num_of_nodes), self.embedding)
        else:
            self.context_embedding = tf.get_variable('context_embedding', [num_of_nodes, dimension],
                                                     initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=num_of_nodes), self.context_embedding)
        self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)


class AliasSampling:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res
