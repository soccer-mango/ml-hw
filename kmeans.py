from cmath import inf
from dis import dis
from logging import NullHandler
from operator import truediv
from re import X
import numpy as np
import random
import math


class Kmeans:
    def __init__(self, X, K, max_iters):
        # Data
        self.X = X
        # Number of clusters
        self.K = K
        # Number of maximum iterations
        self.max_iters = max_iters
        # Initialize centroids
        self.centroids = self.init_centroids()

    def init_centroids(self):
        """
        Selects k random rows from inputs and returns them as the chosen centroids.
        You should randomly choose these rows without replacement and only
        choose from the unique rows in the dataset. Hint: look at
        Python's random.sample function as well as np.unique
        :return: a Numpy array of k cluster centroids, one per row
        """
        # TODO
        return np.array(random.sample(list(np.unique(self.X, axis=1)), self.K))

    def euclidean_dist(self, x, y):
        """
        Computes the Euclidean distance between two points, x and y

        :param x: the first data point, a Python numpy array
        :param y: the second data point, a Python numpy array
        :return: the Euclidean distance between x and y
        """
        # TODO
        return np.linalg.norm(x-y)

    def closest_centroids(self):
        """
        Computes the closest centroid for each data point in X, returning
        an array of centroid indices

        :return: an array of centroid indices
        """
        # TODO

        # List of centroid indices corresponding to each data point in X

        closest_centroids = list()

        for data_point in self.X:
            closest_centroid_idx = None
            closest_centroid_dist = inf

            centroid_index = 0
            while centroid_index < len(self.centroids):
                centroid = self.centroids[centroid_index]
                distance = self.euclidean_dist(data_point, centroid)

                if distance < closest_centroid_dist:
                    closest_centroid_dist = distance
                    closest_centroid_idx = centroid_index
            
                centroid_index += 1

            closest_centroids.append(closest_centroid_idx)

        return np.asarray(closest_centroids)

    def compute_centroids(self, centroid_indices):
        """
        Computes the centroids for each cluster, or the average of all data points
        in the cluster. Update self.centroids.

        Check for convergence (new centroid coordinates match those of existing
        centroids) and return a Boolean whether k-means has converged

        :param centroid_indices: a Numpy array of centroid indices, one for each datapoint in X
        :return boolean: whether k-means has converged
        """
        # TODO
        clusters = np.full(self.centroids.shape, 0.)
        num_points_cluster = list(range(len(clusters)))

        for i in range(len(self.X)):
            data_point = self.X[i]
            centroid_index = centroid_indices[i]

            clusters[centroid_index] = np.add(data_point, clusters[centroid_index])
            num_points_cluster[centroid_index] += 1

        for j in range(len(clusters)):
            clusters[j] = clusters[j]/num_points_cluster[j]
        
        convergence = clusters == self.centroids
        self.centroids = np.array(clusters)

        if convergence.all():
            return True
        else:
            return False

    def run(self):
        """
        Run the k-means algorithm on dataset X with K clusters for max_iters.
        Make sure to call closest_centroids and compute_centroids! Stop early
        if algorithm has converged.
        :return: a tuple of (cluster centroids, indices for each data point)
        Note: cluster centroids and indices should both be numpy ndarrays
        """
        # TODO

        converged = False

        centroid_indices = self.closest_centroids()

        while not converged:
            centroid_indices = self.closest_centroids()
            converged = self.compute_centroids(centroid_indices)
        
        return self.centroids, centroid_indices

    def inertia(self, centroids, centroid_indices):
        """
        Returns the inertia of the clustering. Inertia is defined as the
        sum of the squared distances between each data point and the centroid of
        its assigned cluster.

        :param centroids - the coordinates that represent the center of the clusters
        :param centroid_indices - the index of the centroid that corresponding data point it closest to
        :return inertia as a float
        """
        # TODO
        inertia = 0.0

        i = 0
        while i < len(self.X):
            inertia += self.euclidean_dist(self.X[i], centroids[centroid_indices[i]])

            i += 1
        
        return inertia
