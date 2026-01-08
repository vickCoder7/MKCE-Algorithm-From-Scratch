# Normalized Spectral Clustering
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.cluster import KMeans

def NSC(X, W, k, eps=1e-10):
    """
    Perform Normalized Spectral Clustering on data X with similarity matrix W.

    Args:
        X : ndarray, shape (N, d)
            Dataset
        W : ndarray, shape (N, N)
            Similarity matrix

    Returns:
        labels : ndarray, shape (N,)
            Cluster labels for each data point
    """

    # compute the normalized graph Laplacian
    L = laplacian(W, normed=True)

    # compute the first k eigenvectors of L
    eigvals, eigvecs = eigh(L)
    V = eigvecs[:, :k]

    # normalize rows of V
    V_norm = np.linalg.norm(V, axis=1, keepdims=True)
    V_normalized = V / V_norm
    U = V_normalized.copy()

    # cluster rows of U using k-means
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(U)
    labels = kmeans.labels_

    return labels

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_moons
    import matplotlib.pyplot as plt
    import pandas as pd

    # generate synthetic data
    # X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)
    data =  pd.read_csv("../../datasets/flame.csv", index_col=False)
    data = data.drop(columns=['id', 'z'])
    X = data.to_numpy()

    # construct similarity matrix W using RBF kernel
    from sklearn.metrics.pairwise import rbf_kernel
    W = rbf_kernel(X, gamma=15)

    # perform NSC
    k = 2
    labels = NSC(X, W, k)

    # plot results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title('Normalized Spectral Clustering Results')
    plt.show()
