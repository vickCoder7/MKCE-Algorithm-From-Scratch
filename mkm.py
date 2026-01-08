import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MKM(X, K, eps, Tmax, tol=1e-9, random_state=None):
    """
    Multiple k-Means (MKM) algorithm

    Args:
        X : ndarray, shape (N, d)
            Dataset
        K : list
            k_h values (number of clusters at each iteration h)
        eps : float
            Neighborhood radius ε
        Tmax : int
            Maximum number of iterations
        tol : float
            Numerical tolerance for F < F'
        random_state : int or None
            RNG seed
    Returns:
        Pi : list
            List of partitions π_h (label arrays of size N)
        V : list
            List of centroid sets v_h
        h : int
            Number of iterations performed
        Theta : list
            List of indicator vectors θ_h (arrays of size N)
    """

    rng = np.random.default_rng(random_state)
    N, d = X.shape

    # Initialization
    Pi = []
    V = []
    S_idx = np.arange(N)
    h = 0 
    theta = np.ones(N, dtype=int)  # indicator vector of whether x_i ∈ S'
    Theta = []

    # Distance function
    distance = lambda x, v: np.linalg.norm(x - v)

    # credibility function
    credibility = lambda x, v: distance(x, v) <= eps

    k_h = 2 # fix number of clusters per iteration to 2

    while len(S_idx) >= k_h ** 2 and h < Tmax:
        # initialize variables
        F = 0.0 # objective function
        F_prev = 1.0
        h += 1

        # Randomly select k_h objects from S as initial centroids
        init_idx = rng.choice(S_idx, size=k_h, replace=False)
        v_h = X[init_idx]

        # k-means iteration
        while F < F_prev - tol:
            F_prev = F
            pi_h = np.zeros(N, dtype=int) # cluster assignments

            for i in S_idx:
                distances = [distance(X[i], v_h[l]) for l in range(k_h)]
                pi_h[i] = np.argmin(distances)

            # Update centroids
            for l in range(k_h):
                D = [
                    X[i]
                    for i in S_idx
                    if pi_h[i] == l and credibility(X[i], v_h[l])
                ]
                if len(D) > 0:
                    v_h[l] = np.mean(D, axis=0)

            # Objective function
            F = 0.0
            for l in range(k_h):
                for i in S_idx:
                    if pi_h[i] == l:
                        F += distance(X[i], v_h[l]) ** 2

        # Assign remaining points (X − S)
        for i in range(N):
            if i not in S_idx:
                distances = [distance(X[i], v_h[l]) for l in range(k_h)]
                pi_h[i] = np.argmin(distances)

        # S' = { x_i | π_h(x_i) = 1, x_i ∈ S }
        Sprime_idx = np.array([i for i in S_idx if any(credibility(X[i], v_h[l]) for l in range(k_h))])

        # Update θ
        for i in range(N):
            if i in Sprime_idx:
                theta[i] = 0
            else:
                theta[i] = theta[i]

        # Update outputs and S
        Pi.append(pi_h.copy())
        V.append(v_h.copy())
        Theta.append(theta.copy())

        S_idx = np.array([i for i in S_idx if i not in Sprime_idx])

        # plot current clustering with only points in credibility region
        # plt.figure()
        # plt.scatter(X[:, 0], X[:, 1], color='black', alpha=0.5)
        # for l in range(k_h):
        #     cluster_points = X[[i for i in range(N) if pi_h[i] == l and theta[i] == 0]]
        #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {l+1}', color='lightgrey')
        #     circle = plt.Circle((v_h[l][0], v_h[l][1]), eps, color='r', fill=False, linestyle='--')
        #     plt.gca().add_artist(circle)
        #     plt.scatter(v_h[l][0], v_h[l][1], color='red', marker='x', s=100)
        
        # plt.title(f'MKM Clustering at iteration h={h}')
        # plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.legend()
        # # save the plot
        # # plt.savefig(f"./mkm_images/iteration_{h}.png")
        # plt.show()

    return Pi, V, h, Theta

if __name__ == "__main__":
    data =  pd.read_csv("../../datasets/flame.csv", index_col=False)
    data = data.drop(columns=['id', 'z'])
    X = data.to_numpy()
    eps = 1.5
    N = X.shape[0]

    Pi, V, h, Theta = MKM(X, K=[], eps=eps, Tmax=100, random_state=42)
    print(f"Number of iterations: {h}")