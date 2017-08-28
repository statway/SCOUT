import numpy as np
import pandas as pd

from sklearn import manifold, decomposition
from sklearn import cluster, mixture
from sklearn.metrics.pairwise import pairwise_distances

from scipy import sparse,linalg
from scipy.spatial import distance

import itertools

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use(["seaborn-darkgrid", "seaborn-colorblind", "seaborn-notebook"])


def get_centroid_dists(centroid_i0, centroid_i1, centroid_i2):
    """
    Get the distances between landmark i and landmark i-1,i+1.
    :param centroid_i0: position of landmark i-1;
    :param centroid_i1: position of landmark i;
    :param centroid_i2: position of landmark i+1;
    :return: distances between landmark i and landmark i-1,i+1.
    """
    dist_a = pairwise_distances(centroid_i0, centroid_i1)
    dist_b = pairwise_distances(centroid_i1, centroid_i2)

    return dist_a, dist_b


def get_cell_dists(cluster_values, centroid_i0, centroid_i2):
    """
    Get distances between cluster i and landmark i-1,i+1.
    :param cluster_values: positions of all points for cluster i
    :param centroid_i0: position of landmark i-1;
    :param centroid_i2: position of landmark i+1;
    :return: distances between cluster i and landmark i-1,i+1.
    """
    dist_cell_i0 = pairwise_distances(cluster_values, centroid_i0)
    dist_cell_i2 = pairwise_distances(cluster_values, centroid_i2)
    return dist_cell_i0, dist_cell_i2


class SCOUT():
    """
    The pseudotime ordering of single cells.
    """
    def __init__(self, data, stages=None):
        """
        :param data: N cells * D genes.
        :param stages: time stages of single cells.
        """

        self.data = data
        self.stages = stages

        self.centroids = None
        self.labels = None

        self.Tcsr = None
        self.edges = None

    def dim_down(self, method='tsne', ndim=2, rand_seed=6):
        """
        :param method: selected method of dimension reduction.
        :param ndim: number of retained dimensions.
        :param rand_seed: seed used by the random number generator.
        :return: embedding space with N cells * d feature.
        """

        X = self.data

        # http://scikit-learn.org/stable/modules/manifold.html
        if method == 'tsne' or method == 'TSNE':
            print("Dimension reduction with t-stochastic neighbor embedding(tSNE).\n")
            V = manifold.TSNE(n_components=ndim, random_state=rand_seed, init='pca').fit_transform(X)

        if method == 'lle' or method == 'LLE':
            print("Dimension reduction with locally_linear_embedding(LLE).\n")
            V, err = manifold.locally_linear_embedding(X, n_neighbors=20, n_components=ndim, random_state=rand_seed,
                                                       method='modified')
        if method == 'mds' or method =='MDS':
            print("Dimension reduction with Multidimensional scaling(MDS).\n")
            V = manifold.MDS(n_components=ndim, random_state=rand_seed,max_iter=100, n_init=1).fit_transform(X)

        if method == 'se' or method == 'SE':
            print("Dimension reduction with Spectral Embedding(SE).\n")
            V = manifold.SpectralEmbedding(n_components=ndim, random_state=rand_seed).fit_transform(X)

        # http://scikit-learn.org/stable/modules/decomposition.html
        if method == 'ica' or method == 'ICA':
            print("Matrix decomposition with Independent component analysis(FastICA).\n")
            V = decomposition.FastICA(n_components=ndim, random_state=rand_seed).fit_transform(X)

        if method == 'pca' or method == 'PCA':
            print("Matrix decomposition with Principal component analysis(PCA).\n")
            V = decomposition.PCA(n_components=ndim, random_state=rand_seed).fit_transform(X)

        return V

    def clustering(self, V, method='kmeans', nclust=None, cov = 'tied',rand_seed=6, traj_branch = True):
        """
        :param V: the data for clustering.
        :param method: selected method of clustering.
        :param nclust: number of clusters.
        :param cov: only used for method GMM. Covariance type contains 'full','tied','diag',and 'spherical'.
        :param rand_seed: seed used by the random number generator.
        :return: centroids and labels of the data.
        """

        # http://scikit-learn.org/stable/modules/clustering.html
        if method == 'kmeans' or method == 'Kmeans':
            print("Clustering with K-Means")
            kmeans = cluster.KMeans(n_clusters=nclust, random_state=rand_seed).fit(V)
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_

        if method == 'meanshift':
            print("Clustering with meanshift")
            ms = cluster.MeanShift().fit(V)
            centroids = ms.cluster_centers_
            labels = ms.labels_

        # http://scikit-learn.org/stable/modules/mixture.html
        if method == 'gmm' or method == 'GMM':
            print("Clustering with Gaussian Mixture")
            gmm = mixture.GaussianMixture(n_components=nclust, covariance_type= cov ,random_state=rand_seed).fit(V)
            centroids = gmm.means_
            labels = gmm.predict(V)

        self.centroids = centroids
        self.labels = labels

        if traj_branch == True:
            self.get_MST()
            self.traj_centroids()

        return centroids, labels

    def get_MST(self):
        """
        Minimum spanning tree of centroids.
        :return: MST array and edges.
        """

        centroids = self.centroids
        center_dist = distance.squareform(distance.pdist(centroids))
        matri = sparse.csr_matrix(center_dist)
        Tcsr = sparse.csgraph.minimum_spanning_tree(matri).toarray()
        edges = np.transpose(np.nonzero(Tcsr))

        self.Tcsr = Tcsr
        self.edges = edges

        return Tcsr, edges

    def traj_centroids(self):
        """
        :return: all branches of centroids generated by MST.
        """

        from collections import Counter
        import networkx as nx

        edge_dict = Counter(self.edges.flatten())
        edge_count1 = [edge for edge, count in edge_dict.items() if count == 1]

        traj_centroids = []
        len_edge_count1 = len(edge_count1)

        G = nx.Graph(self.Tcsr)

        for i in (range(len_edge_count1 - 1)):
            for j in range(i + 1, len_edge_count1):
                path = nx.shortest_path(G, source=edge_count1[i], target=edge_count1[j])
                traj_centroids.append(path)

        traj_centroids.sort(key=len, reverse=True)

        print("branches:\n",traj_centroids)

        return traj_centroids

    def get_vertex(self, V, traj_branch, method = '1'):
        """
        :param V: the data.
        :param traj_branch: one single branch for ordering.
        :param method: '1' or '2'.
               '1' : select the cell based on largest distance;
               '2' : select the cell based on projection.
        :return: start cell and end cell of the branch.
        """

        centroid_order = np.asarray([self.centroids[order] for order in traj_branch])

        cluster_start_indices = [i for i, x in enumerate(self.labels) if x == traj_branch[0]]
        cluster_start_values = V[cluster_start_indices, :]

        centroid_1 = np.array(centroid_order[0, :]).reshape((1, -1))
        centroid_2 = np.array(centroid_order[1, :]).reshape((1, -1))

        if method == '1':

            dist_cell_c1 = pairwise_distances(cluster_start_values, centroid_1)
            dist_cell_c2 = pairwise_distances(cluster_start_values, centroid_2)
            dist_cell_max = dist_cell_c1 + 2 * dist_cell_c2

        elif method == '2':

            centroid_vec = centroid_1 - centroid_2
            cluster_start_values = cluster_start_values - centroid_2
            dist_cell_max = cluster_start_values.dot(centroid_vec.transpose())

        start_cell_index = cluster_start_indices[np.argmax(dist_cell_max)]
        branch_start_cell = np.array(V[start_cell_index, :]).reshape((1, -1))

        cluster_end_indices = [i for i, x in enumerate(self.labels) if x == traj_branch[-1]]
        cluster_end_values = V[cluster_end_indices, :]

        centroid_1 = np.array(centroid_order[-1, :]).reshape((1, -1))
        centroid_2 = np.array(centroid_order[-2, :]).reshape((1, -1))

        if method == '1':

            dist_cell_c1 = pairwise_distances(cluster_end_values, centroid_1)
            dist_cell_c2 = pairwise_distances(cluster_end_values, centroid_2)
            dist_cell_max = dist_cell_c1 + dist_cell_c2

        elif method == '2':

            centroid_vec = centroid_1 - centroid_2
            cluster_end_values = cluster_end_values - centroid_2
            dist_cell_max = cluster_end_values.dot(centroid_vec.transpose())

        end_cell_index = cluster_end_indices[np.argmax(dist_cell_max)]
        branch_end_cell = np.array(V[end_cell_index, :]).reshape((1, -1))

        return branch_start_cell, branch_end_cell


    def get_ordering(self, V, traj_branches, method = 'wd', delta = 0.05):
        """
        Key procedure.
        :param V: The data for ordering.
        :param traj_branches: all branches of the data, e.g. [[],[]].
        :param method: 'wd' or 'ap'.
               'wd': ordering by weighted distance.
               'ap': ordering by apollonian circle projection.
        :param delta: scaling factor of method 'wd'. Smaller delta will give a larger weight for near landmark.
        :return: scores of all cells, and cell orders of each branch.
        """

        cell_scores = np.zeros(len(V))
        orders = []

        # adjust edges
        edges = []
        for branch in traj_branches:
            for ind in range(len(branch) - 1):
                edges.append([branch[ind], branch[ind + 1]])

        fset = set(frozenset(edge) for edge in edges)
        edges = [list(x) for x in fset]
        self.edges = edges

        for branch in traj_branches:

            branch_start_cell, branch_end_cell = self.get_vertex(V, branch, method='1')

            centroid_order = np.asarray([self.centroids[order] for order in branch])       # order centroids
            landmark_order = np.vstack((branch_start_cell,centroid_order,branch_end_cell)) # add start_cell and branch_end_cell

            branch_indices = [i for i, x in enumerate(self.labels) if x in branch]
            branch_labels = [x for i, x in enumerate(self.labels) if x in branch]
            branch_values = V[branch_indices, :]

            landmark_mat = distance.squareform(distance.pdist(landmark_order))

            # cells_mat = pairwise_distances(cluster_values,centroid_order)
            cells_mat = distance.cdist(branch_values, landmark_order)

            num_cells = len(branch_values)
            num_landmark = len(landmark_order)

            # store the distance from every landmark to start cell
            center_score = np.zeros(num_landmark)

            for i in range(num_landmark - 1):
                center_score[i + 1] = landmark_mat[i, i + 1] + center_score[i]

            if method == 'wd':

                center_dist = np.zeros([num_landmark, num_landmark])
                for i in range(num_landmark - 1):
                    for j in range(i + 1, num_landmark):  # j>i
                        for k in range(i, j):
                            center_dist[i, j] += landmark_mat[k, k+1]

                P = np.zeros([num_cells, num_landmark])
                W = np.zeros([num_cells, num_landmark])

                for ind, clust in enumerate(branch):

                    clust_indices = [i for i, x in enumerate(branch_labels) if x == clust]

                    for k in range(1, ind+1):

                        P[clust_indices, k] = center_score[k] + cells_mat[clust_indices, k]
                        W[clust_indices, k] = center_dist[k, ind + 1] + cells_mat[clust_indices, ind + 1]

                    for k in range(ind + 2, num_landmark):

                        P[clust_indices, k] = center_score[k] - cells_mat[clust_indices, k]
                        W[clust_indices, k] = center_dist[ind + 1, k] + cells_mat[clust_indices, ind + 1]

                sigma = delta * np.square(W).sum(axis=1) / (num_landmark - 2)

                for ind, clust in enumerate(branch):

                    clust_indices = [i for i, x in enumerate(branch_labels) if x == clust]

                    for k in (itertools.chain(range(1, ind+1), range(ind+2, num_landmark))):

                        W [clust_indices, k] = np.exp(- np.square(W[clust_indices, k]) / sigma[clust_indices])

                W = np.divide(W, W.sum(axis=1)[:,None])
                S = P * W
                cell_score = S.sum(axis=1)
                cell_scores.flat[branch_indices] = cell_score

                branch_order_ind = sorted(range(len(cell_score)), key=lambda k: cell_score[k])
                order = [branch_indices[i] for i in branch_order_ind]

            elif method =='ap':

                branch_scores = np.zeros(len(branch_values))

                pre_dist = 0

                for ind, clust in enumerate(branch):
                    # find cluster data
                    clust_indices = [i for i, x in enumerate(self.labels) if x == clust]
                    clust_values = V[clust_indices, :]

                    # centroid i
                    centroid_i1 = np.array(centroid_order[ind, :]).reshape((1, -1))

                    # dist between centroid i and cluster
                    dist_cell_i1 = pairwise_distances(clust_values, centroid_i1)

                    # find the nearest cell to the centroid_i1
                    centroid_order[ind, :] = clust_values[np.argmin(dist_cell_i1)]

                    # center = centroid_order[ind, :]
                    # for i, value in enumerate(V):
                    #     if np.array_equal(center,value):
                    #         print(i)

                for ind, clust in enumerate(branch):

                    # find cluster data
                    clust_indices = [i for i, x in enumerate(self.labels) if x == clust]
                    branch_clust_indices = [i for i, x in enumerate(branch_labels) if x == clust]
                    clust_values = V[clust_indices, :]

                    # centroid i
                    centroid_i1 = np.array(centroid_order[ind, :]).reshape((1, -1))

                    # centroid i-1 and i+1
                    if ind == 0:
                        centroid_i0 = branch_start_cell
                        centroid_i2 = np.array(centroid_order[ind + 1, :]).reshape((1, -1))

                    elif 0 < ind < (len(branch) - 1):
                        centroid_i0 = np.array(centroid_order[ind - 1, :]).reshape((1, -1))
                        centroid_i2 = np.array(centroid_order[ind + 1, :]).reshape((1, -1))

                    elif ind == (len(branch) - 1):
                        centroid_i0 = np.array(centroid_order[ind - 1, :]).reshape((1, -1))
                        centroid_i2 = branch_end_cell

                    # dist between cell and centroid i-1,i+1
                    dist_cell_i0, dist_cell_i2 = get_cell_dists(clust_values, centroid_i0,  centroid_i2)

                    # dist between centroids
                    dist_a, dist_b = get_centroid_dists(centroid_i0, centroid_i1, centroid_i2)

                    dist_cell_i02 = dist_cell_i0 + dist_cell_i2
                    dist_ab = dist_a + dist_b
                    cell_score = (dist_cell_i0 / dist_cell_i02) * dist_ab + pre_dist

                    pre_dist = pre_dist + dist_a

                    cell_scores.flat[clust_indices] = cell_score
                    branch_scores.flat[branch_clust_indices]=cell_score

                branch_order_ind = sorted(range(len(branch_scores)), key=lambda k: branch_scores[k])
                order = [branch_indices[i] for i in branch_order_ind]

            orders.append(order)
        # orders = sorted(range(len(cell_scores)), key=lambda k: cell_scores[k])
        return cell_scores, orders

    def plotD(self,V,dims=[0,1]):
        """
        Plot the embedding structure of the data.
        """

        if V.shape[1] < len(dims):
            raise Exception('The number of dimensions for plots should be no greater than dimensions for embedding space.')

        fig = plt.figure(figsize=(25, 20))

        if len(dims) == 2:
            if V.shape[1] > 2:
                print("When the number of embedding space dimensions is more than 3, you could set 3D plot, e.g. dims = [0,1,2]")

            if self.labels is not None:
                plt.scatter(V[:, dims[0]], V[:, dims[1]], c=np.array(self.labels), cmap='jet')
                centroids = self.centroids
                plt.scatter(centroids[:, dims[0]], centroids[:, dims[1]], marker='o', s=100)
                for index, centroid in enumerate(centroids):
                    plt.annotate(index, xy=(centroid[dims[0]], centroid[dims[1]]), fontsize=30)

                if self.edges is not None:
                    for edge in self.edges:
                        i, j = edge
                        plt.plot([centroids[i, dims[0]], centroids[j, dims[0]]], [centroids[i, dims[1]], centroids[j, dims[1]]], linewidth=0.2,
                                 c='r')

            elif self.stages is not None:
                time_stage = sorted(set(self.stages))
                color_stage = cm.jet(np.linspace(0, 1, len(time_stage)))
                color_dict = dict(zip(time_stage, color_stage[:len(time_stage)]))
                cell_color = [color_dict[cst] for cst in self.stages]
                markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in
                           color_dict.values()]

                for v, c in zip(V, cell_color):
                    plt.scatter(v[dims[0]], v[dims[1]], color=c)

                plt.xlabel('D'+str(dims[0]),fontsize=15,fontweight='bold')
                plt.ylabel('D'+str(dims[1]),fontsize=15,fontweight='bold')

                plt.legend(markers, color_dict.keys(), numpoints=3)

            else:
                plt.scatter(V[:, dims[0]], V[:, dims[1]])

        elif len(dims) == 3:

            ax = fig.add_subplot(111, projection='3d')

            if self.labels is not None:
                ax.scatter(V[:, dims[0]], V[:, dims[1]], V[:, dims[2]], c=np.array(self.labels), cmap='jet')
                centroids = self.centroids
                ax.scatter(centroids[:, dims[0]], centroids[:, dims[1]], centroids[:, dims[2]], marker='o', s=100)
                for index in range(len(centroids)):
                    ax.text(centroids[index, dims[0]], centroids[index, dims[1]], centroids[index, dims[2]], '%s' % (str(index)), size=30)

                if self.edges is not None:
                    for edge in self.edges:
                        i, j = edge
                        ax.plot([centroids[i, dims[0]], centroids[j, dims[0]]], [centroids[i, dims[1]], centroids[j, dims[1]]],
                                [centroids[i, dims[2]], centroids[j, dims[2]]], linewidth=0.2, c='r')

            elif self.stages is not None:
                time_stage = sorted(set(self.stages))
                color_stage = cm.jet(np.linspace(0, 1, len(time_stage)))
                color_dict = dict(zip(time_stage, color_stage[:len(time_stage)]))
                cell_color = [color_dict[cst] for cst in self.stages]
                markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in
                           color_dict.values()]

                for v, c in zip(V, cell_color):
                    ax.scatter(v[dims[0]], v[dims[1]], v[dims[2]], color=c)

                ax.grid(True)
                # ax.set_xticklabels([])
                # ax.set_yticklabels([])
                # ax.set_zticklabels([])

                ax.set_xlabel('D'+str(dims[0]),fontsize=15,fontweight='bold')
                ax.set_ylabel('D'+str(dims[1]),fontsize=15,fontweight='bold')
                ax.set_zlabel('D'+str(dims[2]),fontsize=15,fontweight='bold')

                ax.legend(markers, color_dict.keys(), loc=2, numpoints=3)

            else:
                ax.scatter(V[:, dims[0]], V[:, dims[1]], V[:, dims[2]])

        plt.show()

    def plotT(self, V, cell_scores, dims = [0,1], plotlandmark = True):
        """
        Plot the inferred trajectory in the embedding structure of the data.
        """

        fig = plt.figure(figsize=(25, 20))
        if len(dims) == 2:
            if V.shape[1] > 2:
                print("When the number of embedding space dimensions is more than 3, you could set 3D plot, e.g. dims = [0,1,2]")

            p = plt.scatter(V[:, dims[0]], V[:, dims[1]], c=cell_scores, cmap='jet')

            if plotlandmark == True:
                centroids = self.centroids
                plt.scatter(centroids[:, dims[0]], centroids[:, dims[1]], marker='o', s=100)

                for edge in self.edges:
                    i, j = edge
                    plt.plot([centroids[i, dims[0]], centroids[j, dims[0]]], [centroids[i, dims[1]], centroids[j, dims[1]]], c='r')

            plt.colorbar(p)

        elif len(dims) ==3:
            ax = fig.add_subplot(111, projection='3d')
            p = ax.scatter(V[:, dims[0]], V[:, dims[1]], V[:, dims[2]], c=cell_scores, cmap='jet')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            ax.set_xlabel('D' + str(dims[0]), fontsize=15, fontweight='bold')
            ax.set_ylabel('D' + str(dims[1]), fontsize=15, fontweight='bold')
            ax.set_zlabel('D' + str(dims[2]), fontsize=15, fontweight='bold')

            if plotlandmark == True:
                centroids = self.centroids
                ax.scatter(centroids[:, dims[0]], centroids[:, dims[1]], centroids[:, dims[2]], marker='o', s=100)

                for edge in self.edges:
                    i, j = edge
                    ax.plot([centroids[i, dims[0]], centroids[j, dims[0]]], [centroids[i, dims[1]], centroids[j, dims[1]]],
                            [centroids[i, dims[2]], centroids[j, dims[2]]], c='r')
            plt.colorbar(p)

        plt.show()


if __name__ == "__main__":
    pass
