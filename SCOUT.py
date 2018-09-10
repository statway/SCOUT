import numpy as np
import pandas as pd

from sklearn import manifold, decomposition
from sklearn import cluster, mixture
from sklearn.metrics.pairwise import pairwise_distances

from scipy import sparse,linalg,spatial
from scipy.spatial import distance

import itertools

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

plt.style.use(["seaborn-darkgrid", "seaborn-colorblind", "seaborn-notebook"])


def get_landmark_dists(landmark_i0, landmark_i1, landmark_i2):
    """
    Get the distances between landmark i and landmark i-1,i+1.
    :param landmark_i0: position of landmark i-1;
    :param landmark_i1: position of landmark i;
    :param landmark_i2: position of landmark i+1;
    :return: distances between landmark i and landmark i-1,i+1.
    """
    # spatial.distance.cdist
    dist_a = pairwise_distances(landmark_i0, landmark_i1)
    dist_b = pairwise_distances(landmark_i1, landmark_i2)

    return dist_a, dist_b


def get_cell_dists(cluster_values, landmark_i0, landmark_i2):
    """
    Get distances between cluster i and landmark i-1,i+1.
    :param cluster_values: positions of all points for cluster i
    :param landmark_i0: position of landmark i-1;
    :param landmark_i2: position of landmark i+1;
    :return: distances between cluster i and landmark i-1,i+1.
    """
    # spatial.distance.cdist
    dist_cell_i0 = pairwise_distances(cluster_values, landmark_i0)
    dist_cell_i2 = pairwise_distances(cluster_values, landmark_i2)
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

        self.landmarks = None
        self.labels = None
        self.traj_sort_ind = None
        self.landmark_indices = None
        self.landmarks_sort = None

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

    def cluster_landmarks(self, V, method='GMM', nclust=None, cov = 'tied',rand_seed=6, traj_branch = True):
        """
        :param V: the data for clustering.
        :param method: selected method of clustering.
        :param nclust: number of clusters.
        :param cov: only used for method GMM. Covariance type contains 'full','tied','diag',and 'spherical'.
        :param rand_seed: seed used by the random number generator.
        :return: landmarks and labels of the data.
        """

        # http://scikit-learn.org/stable/modules/clustering.html
        if method == 'kmeans' or method == 'Kmeans':
            print("Clustering with K-Means")
            kmeans = cluster.KMeans(n_clusters=nclust, random_state=rand_seed).fit(V)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_

        if method == 'meanshift':
            print("Clustering with meanshift")
            ms = cluster.MeanShift().fit(V)
            centers = ms.cluster_centers_
            labels = ms.labels_

        # http://scikit-learn.org/stable/modules/mixture.html
        if method == 'gmm' or method == 'GMM':
            print("Clustering with Gaussian Mixture")
            if nclust == None:
                lowest_bic = np.infty
                bic = []
                n_components_range = range(2, 20)
                cv_types = ['spherical', 'tied', 'diag', 'full']
                for cv_type in cv_types:
                    for n_components in n_components_range:
                        # Fit a Gaussian mixture with EM
                        gmm = mixture.GaussianMixture(n_components=n_components,
                                                      covariance_type=cv_type)
                        gmm.fit(V)
                        bic.append(gmm.bic(V))
                        if bic[-1] < lowest_bic:
                            lowest_bic = bic[-1]
                            best_gmm = gmm
                centers = best_gmm.means_
                labels = best_gmm.predict(V)
            else:
                gmm = mixture.GaussianMixture(n_components=nclust, covariance_type= cov ,random_state=rand_seed).fit(V)
                centers = gmm.means_
                labels = gmm.predict(V)
        
        from sklearn.metrics import pairwise_distances_argmin_min
        # find nearest cell as landmarks
        landmark_indices, _ = pairwise_distances_argmin_min(centers, V) 
        
        self.landmark_indices  = landmark_indices
        self.landmarks = V[landmark_indices,:]
        self.labels = labels

        if traj_branch == True:
            self.get_traj_landmarks(V)

        return self.landmarks

    def density_landmarks(self,V, r = None, delta = 0.5,traj_branch = True):

        N = V.shape[0]
        kdtree = spatial.cKDTree(V)
        bsig = np.sum([kdtree.query(V[i,:],k=N)[0][N-1] for i in range(N)])/N
        ssig = np.sum([kdtree.query(V[i,:],k=2)[0][1] for i in range(N)])/N
        if r == None:
            import math
            r = ssig * math.log(bsig/ssig,2)
            print("Fixed-radius is ",r)

        ball_neighbor_list = []
        density_neighbor_list = []
        landmark_indices = []
        sorted_indices = []

        for item in V:
            ball_neighbor = kdtree.query_ball_point(item, r)
            ball_neighbor_list.append(ball_neighbor)

        len_neighbor_list = [len(item) for item in ball_neighbor_list]
        
        for index,item in enumerate(ball_neighbor_list):
            a = np.array(V[index,:].reshape((1, -1)))
            b = np.array(V[item,:].reshape((-1, V.shape[1])))
            ab_dist = spatial.distance.cdist(a,b)
            if len(ab_dist[0]) == 1:
                sigma = 1
            else:
                sigma = np.std(ab_dist)
            density_neighbor_list.append(np.sum(np.exp(- (1/sigma) * (ab_dist))))

        sorted_indices = np.argsort(density_neighbor_list)[::-1]
        indices_set = set(sorted_indices)


        for ind in sorted_indices:

            if density_neighbor_list[ind] < delta * density_neighbor_list[sorted_indices[0]]:
                break
            if ind not in indices_set:
                continue
            else:
                landmark_indices.append(ind)
                indices_set = indices_set - set(ball_neighbor_list[ind])
        
        self.landmark_indices  = landmark_indices
        self.landmarks = V[landmark_indices,:]

        cell_labels = []
        for i in range(V.shape[0]):
            cell_label = np.argmin(spatial.distance.cdist(self.landmarks,V[i,:].reshape((1, -1))))
            cell_labels.append(cell_label)
        self.labels = cell_labels
        
        if traj_branch == True:
            self.get_traj_landmarks(V)
        
        return  V[landmark_indices,:]
    
    def get_traj_landmarks(self,V):

        from collections import Counter
        import networkx as nx

        landmarks = self.landmarks
        center_dist = distance.squareform(distance.pdist(landmarks))
        matri = sparse.csr_matrix(center_dist)
        Tcsr = sparse.csgraph.minimum_spanning_tree(matri).toarray()
        edges = np.transpose(np.nonzero(Tcsr))

        self.Tcsr = Tcsr
        self.edges = edges


        landmarks_count_dict = Counter(edges.flatten()) # count numbers of landmarks
        landmarks_ind_count1 = [ind for ind, count in landmarks_count_dict.items() if count == 1] # get vertex indices of MST
        landmark_vertexs = landmarks[landmarks_ind_count1,:].reshape((-1, V.shape[1]))
        
        if self.stages is None:
            start_point = V[0,:].reshape((1, -1))
            landmark_start_ind = np.argmin(spatial.distance.cdist(landmark_vertexs,start_point))
            landmark_ind_1 = landmarks_ind_count1[landmark_start_ind] # get start landmark index
            landmark_ind_M = [item for item in landmarks_ind_count1 if item!=landmark_ind_1]# get end landmarks index

        else:
            landmarks_stages = np.array(self.stages)[self.landmark_indices]
            
            
            landmark_count1_stage = np.array(landmarks_stages)[landmarks_ind_count1]
            min_stage = np.min(landmark_count1_stage)
            min_landmarks_stage_ind =  [ind for ind,value in enumerate(landmark_count1_stage) if value == min_stage]

            if len(min_landmarks_stage_ind) == 1:
                landmark_ind_1 = int(np.array(landmarks_ind_count1)[min_landmarks_stage_ind])
            else:
                print("Start landmark have more than one, need to be checked.")
                landmark_ind_1 = int(np.array(landmarks_ind_count1)[min_landmarks_stage_ind[0]])
            
            landmark_ind_M = [item for item in landmarks_ind_count1 if item!=landmark_ind_1]# get end landmarks index
            
            
        landmark_ind_M = [item for item in landmarks_ind_count1 if item!=landmark_ind_1]# get end landmarks index

        traj_lanmarks_ind = []
        len_landmarkM_count = len(landmark_ind_M)

        G = nx.Graph(Tcsr)

        for j in range(len_landmarkM_count):
            path = nx.shortest_path(G, source=landmark_ind_1, target=landmark_ind_M[j])
            traj_lanmarks_ind.append(path)
        
        traj_lanmarks_ind.sort(key=len, reverse=True) # sort traj ind by length

        traj_sort_ind = traj_lanmarks_ind[0]
        traj_sorted_indices = []
        traj_sorted_indices.append(list(range(len(traj_sort_ind))))
        for i in range(len(traj_lanmarks_ind)-1):
            ext_ind_len = len(set(traj_lanmarks_ind[i+1])-set(traj_sort_ind))
            ext_indices = traj_lanmarks_ind[i+1][-ext_ind_len:]
            old_indices = traj_lanmarks_ind[i+1][:-ext_ind_len or None]
            another_sorted_indices = [i for i,v in enumerate(traj_sort_ind) if v in old_indices]
            list_sorted_indices = list(range(len(traj_sort_ind),len(traj_sort_ind)+ext_ind_len))
            another_sorted_indices.extend(list_sorted_indices)
            traj_sorted_indices.append(another_sorted_indices)
            traj_sort_ind.extend(ext_indices)
        
        print("traj_sorted_indices",traj_sorted_indices)
            
        self.traj_sort_ind = traj_sort_ind
        self.landmarks_sort = landmarks[traj_sort_ind,:]
        traj_branch = True
        return sorted(traj_sort_ind)

    def get_vertex(self, V, traj_branch, method = '1'):
        """
        :param V: the data.
        :param traj_branch: one single branch for ordering.
        :param method: '1' or '2'.
               '1' : select the cell based on largest distance;
               '2' : select the cell based on projection.
        :return: start cell and end cell of the branch.
        """
        # print("traj_branch:",traj_branch)
        # print("landmarks",self.landmarks)

        landmark_order = np.asarray([self.landmarks[order] for order in traj_branch])

        cluster_start_indices = [i for i, x in enumerate(self.labels) if x == traj_branch[0]]
        cluster_start_values = V[cluster_start_indices, :]

        landmark_1 = np.array(landmark_order[0, :]).reshape((1, -1))
        landmark_2 = np.array(landmark_order[1, :]).reshape((1, -1))

        if method == '1':

            dist_cell_c1 = pairwise_distances(cluster_start_values, landmark_1)
            dist_cell_c2 = pairwise_distances(cluster_start_values, landmark_2)
            dist_cell_max = dist_cell_c1 + 2 * dist_cell_c2

        elif method == '2':

            landmark_vec = landmark_1 - landmark_2
            cluster_start_values = cluster_start_values - landmark_2
            dist_cell_max = cluster_start_values.dot(landmark_vec.transpose())

        start_cell_index = cluster_start_indices[np.argmax(dist_cell_max)]
        branch_start_cell = np.array(V[start_cell_index, :]).reshape((1, -1))

        cluster_end_indices = [i for i, x in enumerate(self.labels) if x == traj_branch[-1]]
        cluster_end_values = V[cluster_end_indices, :]

        landmark_1 = np.array(landmark_order[-1, :]).reshape((1, -1))
        landmark_2 = np.array(landmark_order[-2, :]).reshape((1, -1))

        if method == '1':

            dist_cell_c1 = pairwise_distances(cluster_end_values, landmark_1)
            dist_cell_c2 = pairwise_distances(cluster_end_values, landmark_2)
            dist_cell_max = dist_cell_c1 + dist_cell_c2

        elif method == '2':

            landmark_vec = landmark_1 - landmark_2
            cluster_end_values = cluster_end_values - landmark_2
            dist_cell_max = cluster_end_values.dot(landmark_vec.transpose())

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

        traj_sort_ind = self.traj_sort_ind

        # print("traj_sort_ind",traj_sort_ind)
        # print("traj_branches",traj_branches)

        traj_new_indices = []
        for i in range(len(traj_branches)):
            traj_new_indices.append([traj_sort_ind[ind] for ind in traj_branches[i]])
        # print(traj_new_indices)
        
        traj_branches = traj_new_indices
        # adjust edges
        edges = []
        for branch in traj_branches:
            for ind in range(len(branch) - 1):
                edges.append([branch[ind], branch[ind + 1]])

        fset = set(frozenset(edge) for edge in edges)
        edges = [list(x) for x in fset]
        self.edges = edges

        # print("error")

        for branch in traj_branches:

            branch_start_cell, branch_end_cell = self.get_vertex(V, branch, method='1')

            landmark_order = np.asarray([self.landmarks[order] for order in branch])       # order landmarks
            landmark_order = np.vstack((branch_start_cell,landmark_order,branch_end_cell)) # add start_cell and branch_end_cell

            branch_indices = [i for i, x in enumerate(self.labels) if x in branch]
            branch_labels = [x for i, x in enumerate(self.labels) if x in branch]
            branch_values = V[branch_indices, :]

            landmark_mat = distance.squareform(distance.pdist(landmark_order))

            # cells_mat = pairwise_distances(cluster_values,landmark_order)
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

                    # landmark i
                    landmark_i1 = np.array(landmark_order[ind, :]).reshape((1, -1))

                    # dist between landmark i and cluster
                    dist_cell_i1 = pairwise_distances(clust_values, landmark_i1)

                    # find the nearest cell to the landmark_i1
                    landmark_order[ind, :] = clust_values[np.argmin(dist_cell_i1)]

                for ind, clust in enumerate(branch):

                    # find cluster data
                    clust_indices = [i for i, x in enumerate(self.labels) if x == clust]
                    branch_clust_indices = [i for i, x in enumerate(branch_labels) if x == clust]
                    clust_values = V[clust_indices, :]

                    # landmark i
                    landmark_i1 = np.array(landmark_order[ind, :]).reshape((1, -1))

                    # landmark i-1 and i+1
                    if ind == 0:
                        landmark_i0 = branch_start_cell
                        landmark_i2 = np.array(landmark_order[ind + 1, :]).reshape((1, -1))

                    elif 0 < ind < (len(branch) - 1):
                        landmark_i0 = np.array(landmark_order[ind - 1, :]).reshape((1, -1))
                        landmark_i2 = np.array(landmark_order[ind + 1, :]).reshape((1, -1))

                    elif ind == (len(branch) - 1):
                        landmark_i0 = np.array(landmark_order[ind - 1, :]).reshape((1, -1))
                        landmark_i2 = branch_end_cell

                    # dist between cell and landmark i-1,i+1
                    dist_cell_i0, dist_cell_i2 = get_cell_dists(clust_values, landmark_i0,  landmark_i2)

                    # dist between landmarks
                    dist_a, dist_b = get_landmark_dists(landmark_i0, landmark_i1, landmark_i2)

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
        
        landmarks = self.landmarks
        traj_sort_ind = self.traj_sort_ind 

        if len(dims) == 2:
            fig = plt.figure(figsize=(25, 20))
            # plt.axes().set_aspect('equal', 'datalim')
            plt.gca().set_aspect('equal', adjustable='box')

            if V.shape[1] > 2:
                print("When the number of embedding space dimensions is more than 3, you could set 3D plot, e.g. dims = [0,1,2]")

            if self.labels is not None:
                plt.scatter(V[:, dims[0]], V[:, dims[1]], c=np.array(self.labels), cmap='jet',s=30)      
                # print(traj_sort_ind)
                plt.scatter(landmarks[:, dims[0]], landmarks[:, dims[1]], marker='o', s=100)
                for index, landmark in enumerate(landmarks[traj_sort_ind,:]):
                    plt.annotate(index, xy=(landmark[dims[0]], landmark[dims[1]]), fontsize=15)
                    # plt.annotate(index, xy=(landmark[dims[0]], landmark[dims[1]]),fontsize=30)

                if self.edges is not None:
                    for edge in self.edges:
                        i, j = edge
                        plt.plot([landmarks[i, dims[0]], landmarks[j, dims[0]]], [landmarks[i, dims[1]], landmarks[j, dims[1]]], linewidth=0.2,
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

            # ax = fig.add_subplot(111, projection='3d')
            # ax =  plt.axes(projection='3d',figsize=(25,20))
            fig = plt.figure(figsize=(25,20))
            ax = fig.gca(projection='3d')

            if self.labels is not None:
                ax.scatter(V[:, dims[0]], V[:, dims[1]], V[:, dims[2]], c=np.array(self.labels), cmap='jet')
                landmarks = self.landmarks
                ax.scatter(landmarks[:, dims[0]], landmarks[:, dims[1]], landmarks[:, dims[2]], marker='o', s=100)
                
                for index, landmark in enumerate(landmarks[traj_sort_ind,:]):
                    # plt.annotate(index, xy=(landmark[dims[0]], landmark[dims[1]]), fontsize=15)
                    ax.text(landmark[dims[0]], landmark[dims[1]], landmark[dims[2]], '%s' % (str(index)), size=30)
                # for index in range(len(landmarks)):
                #     ax.text(landmarks[index, dims[0]], landmarks[index, dims[1]], landmarks[index, dims[2]], '%s' % (str(index)), size=30)

                if self.edges is not None:
                    for edge in self.edges:
                        i, j = edge
                        ax.plot([landmarks[i, dims[0]], landmarks[j, dims[0]]], [landmarks[i, dims[1]], landmarks[j, dims[1]]],
                                [landmarks[i, dims[2]], landmarks[j, dims[2]]], linewidth=0.2, c='r')

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

        if len(dims) == 2:
            fig = plt.figure(figsize=(25, 20))
            if V.shape[1] > 2:
                print("When the number of embedding space dimensions is more than 3, you could set 3D plot, e.g. dims = [0,1,2]")

            p = plt.scatter(V[:, dims[0]], V[:, dims[1]], c=cell_scores, cmap='jet')

            if plotlandmark == True:
                landmarks = self.landmarks
                plt.scatter(landmarks[:, dims[0]], landmarks[:, dims[1]], marker='o', s=100)

                for edge in self.edges:
                    i, j = edge
                    plt.plot([landmarks[i, dims[0]], landmarks[j, dims[0]]], [landmarks[i, dims[1]], landmarks[j, dims[1]]], c='r')

            plt.colorbar(p)

        elif len(dims) ==3:
            # ax = fig.add_subplot(111, projection='3d')
            # ax =  plt.axes(projection='3d')
            fig = plt.figure(figsize=(25,20))
            ax = fig.gca(projection='3d')

            p = ax.scatter(V[:, dims[0]], V[:, dims[1]], V[:, dims[2]], c=cell_scores, cmap='jet')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            ax.set_xlabel('D' + str(dims[0]), fontsize=15, fontweight='bold')
            ax.set_ylabel('D' + str(dims[1]), fontsize=15, fontweight='bold')
            ax.set_zlabel('D' + str(dims[2]), fontsize=15, fontweight='bold')

            if plotlandmark == True:
                landmarks = self.landmarks
                ax.scatter(landmarks[:, dims[0]], landmarks[:, dims[1]], landmarks[:, dims[2]], marker='o', s=100)

                for edge in self.edges:
                    i, j = edge
                    ax.plot([landmarks[i, dims[0]], landmarks[j, dims[0]]], [landmarks[i, dims[1]], landmarks[j, dims[1]]],
                            [landmarks[i, dims[2]], landmarks[j, dims[2]]], c='r')
            plt.colorbar(p)

        plt.show()


if __name__ == "__main__":
    pass
