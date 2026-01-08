import numpy as np
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from mkm import MKM
from nsc import NSC
import matplotlib.pyplot as plt

def distance(x, v):
    return np.linalg.norm(x - v)

def credibility(x, v, eps):
    return distance(x, v) <= eps

def similarity(X, V, eps):
    """
    Compute similarity matrix W between all clusters in base clusterings.
    According to Eq. (7) in the paper:
    δ(C_hl, C_gj) = |B((v_hl + v_gj)/2)| / d(v_hl, v_gj)  if d(v_hl, v_gj) <= 4*eps
                  = 0                                      otherwise
    
    Args:
        X : ndarray, shape (N, d) - data points
        V : list of ndarrays - cluster centers from all base clusterings
        eps : float - neighborhood radius
    
    Returns:
        W : ndarray, shape (total_clusters, total_clusters), similarity matrix
    """
    # Flatten all cluster centers into a single list
    all_centers = []
    for v_h in V:
        for v_hl in v_h:
            all_centers.append(v_hl)
    all_centers = np.array(all_centers)
    
    num_clusters = len(all_centers)
    W = np.zeros((num_clusters, num_clusters))
    
    for i in range(num_clusters):
        for j in range(num_clusters):
            dist = distance(all_centers[i], all_centers[j])
            if dist <= 4 * eps and dist > 0:
                latent_cluster_center = (all_centers[i] + all_centers[j]) / 2
                num_latent_credible_pts = 0
                for x in X:
                    if credibility(x, latent_cluster_center, eps):
                        num_latent_credible_pts += 1
                W[i, j] = num_latent_credible_pts / dist
            else:
                W[i, j] = 0
            
    return W

def build_cluster_index(Pi):
    """
    Assign a unique global index to each cluster C_{h,l}
    """
    cluster_to_index = {}
    index_to_cluster = {}
    idx = 0

    for h, pi_h in enumerate(Pi):
        for l in np.unique(pi_h):
            cluster_to_index[(h, l)] = idx
            index_to_cluster[idx] = (h, l)
            idx += 1

    return cluster_to_index, index_to_cluster

def relabel_base_clusterings(Pi, cluster_to_index, labels_partition):
    """
    Relabel base clusterings using NSC cluster partition

    Args:
        Pi : list of ndarray
            Pi[h][i] = local cluster label of x_i in clustering h
        cluster_to_index : dict
            (h, local_label) -> cluster_index
        labels_partition : ndarray
            labels_partition[c] = global cluster label assigned by NSC

    Returns:
        R : list of ndarray
            Relabeled base clustering set
    """
    R = []

    for h, pi_h in enumerate(Pi):
        Rh = np.zeros_like(pi_h)

        for i in range(len(pi_h)):
            local_label = pi_h[i]
            cluster_idx = cluster_to_index[(h, local_label)]
            Rh[i] = labels_partition[cluster_idx]

        R.append(Rh)

    return R       
    
def KMCE(X, eps=1.5, Tmax=100, k=2, K=None, tol=1e-9, random_state=None):
    """
    Multiple k-Means Clustering Ensemble (KMCE) algorithm

    Args:
        X : ndarray, shape (N, d)
            Dataset
        eps : float
            Neighborhood radius ε
        Tmax : int
            Maximum number of iterations
        k : int
            Number of clusters to form
        tol : float
            Numerical tolerance
        random_state : int or None
            RNG seed
    Returns:
        Pi_final : list
            Final partition π (label array of size N)
        
    """
    # Generate base clusterings using MKM
    Pi, V = MKM(X, K, eps, Tmax, tol, random_state=random_state)[0:2]
    
    # credibility matrix
    lambda_ = np.zeros((X.shape[0], len(Pi)))

    for i in range(X.shape[0]):
        for h in range(len(Pi)):
            cluster_idx = np.where(Pi[h] == Pi[h][i])[0]
            v_h = V[h][Pi[h][i]]
            if credibility(X[i], v_h, eps):
                lambda_[i, h] = 1
            else:
                lambda_[i, h] = 0

    # similarity matrix between clusters 
    W = similarity(X, V, eps)

    # normalized spectral clustering on cluster similarity graph to partition clusters
    labels_partition = NSC(X, W, k)

    # assign unique cluster index to each cluster
    cluster_to_index, index_to_cluster = build_cluster_index(Pi)

    # relabel base clusterings
    R = relabel_base_clusterings(Pi, cluster_to_index, labels_partition)

    Pi_final = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        votes = np.zeros(k)

        for h in range(len(R)):
            if lambda_[i, h] == 1:
                votes[R[h][i]] += 1

        Pi_final[i] = np.argmax(votes)

    return Pi_final

if __name__ == "__main__":
    data =  pd.read_csv("../../datasets/flame.csv", index_col=False)
    data = data.drop(columns=['id', 'z'])
    X = data.to_numpy()

    # perform KMCE
    eps = 0.5
    k = 2
    Tmax = 100
    Pi_final = KMCE(X, eps=eps, k=k, Tmax=Tmax, tol=1e-9, random_state=42)
    
    print(f"\nRunning KMCE with:")
    print(f"  eps = {eps}")
    print(f"  Tmax = {Tmax}")
    print(f"  k = {k}")

    # plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='gray', s=30)
    plt.title('Data Points')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=Pi_final, cmap='viridis', s=30)
    plt.title('KMCE Clustering Results')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('./kmce_images/kmce_result3.png')
    plt.show()