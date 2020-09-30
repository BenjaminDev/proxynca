import sklearn.cluster
import sklearn.metrics.cluster
import numpy as np
import torch


def assign_by_euclidian_at_k(X, T, k):
    """
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    distances = torch.cdist(X, X)
    # get nearest points
    indices = distances.topk(k + 1, largest=False)[1][:, 1 : k + 1]
    return T[indices]


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1.0 * len(T))


def cluster_by_kmeans(X, nb_clusters):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
    return sklearn.cluster.KMeans(nb_clusters).fit(X).labels_


def calc_normalized_mutual_information(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys)
