
from typing import Optional, Union
import warnings

import numpy as np
import bottleneck as bn

from scipy.sparse import csr_matrix

from sklearn.metrics.pairwise import cosine_similarity

from recpack.algorithms.base import ItemSimilarityMatrixAlgorithm, Algorithm

class BNSLIM(ItemSimilarityMatrixAlgorithm):
    """
    A python impementation of the BNSLIM (Balanced Neighborhood SLIM) algorithm. 
    It is based on the implementation of the authors in LibRec.
    Specifically, it parallels the Java-based BLNSLIMFastRecommender.java file in LibRec.
    Only the knns are considered on each CD update step.
    -----------
    References:
    Burke, Robin, et al. Balanced neighborhoods for multi-sided fairness in recommendation. FAccT 2018
    Sonboli, Nasim, et al. Fairness-aware Recommendation with librec-auto. RecSys 2020
    """
    
    def __init__(self, knn=50, l1=0.5, l2=5, l3=1e3, thr=1e-4, maxIter=10, method="item", seed=None):
        """
        Initializes parameters for BNSLIM.
        -------
        Args:
            knn (int): Item's nearest neighbors for kNN > 0.
            l1 (float): Sparsity-inducing regularizer.
            l2 (float): Overfitting control regularizer.
            l3 (float): Balanced regularizer.
            thr (float): Stopping threshold for convergence.
            maxIter (int): Maximum number of iterations for the optimizer.
            method (str): Specifies whether to use item or user neighborhoods.
        """
        super().__init__()
        self.knn = knn
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.thr = thr
        self.maxIter = maxIter
        self.seed = seed
        self.method = method

    def _fit(self, X: csr_matrix, p: np.ndarray, W_ini: Optional[np.ndarray] = None):
        """
        Fit the BNSLIM model to the provided interaction data.
        -------
        Args:
            X (csr_matrix): A user-item interaction matrix to fit the model to.
            p (np.ndarray): A vector indicating the group membership of each user/item.
                The membership is either 1 or -1, 1 for the users/items belonging to the protected group and -1 for the users/items belonging to the non-protected group.
            W_ini (Optional[np.ndarray]): A precomputed item-item or user-user coefficient matrix that is used as initial guess. 
                If None, the algorithm initializes a matrix filled with small values.
        """

        # Compute the co-occurrence matrix
        C = X.dot(X.T).toarray()
        
        # Set diagonal elements to 0
        np.fill_diagonal(C, 0)

        # Compute the cosine similarity matrix
        S = cosine_similarity(C)

        nearestNeighborCollection = {}
        # Loop over each item/user in the rating matrix
        for i in range(X.shape[0]):
            # Find the n most similar items/users to item/user i
            top_n_idx = np.argsort(S[:, i])[::-1][:self.knn]
            top_n_items = top_n_idx[top_n_idx != i]
            # Store the top n item/user ids as values for the key i in the dictionary
            nearestNeighborCollection[i] = top_n_items.tolist()

        # Initialize the similarity_matrix_ if not given, make small guesses (e.g., W.ini(0.01)) to speed up the training
        if W_ini is None: 
            rng = np.random.default_rng(self.seed)
            W_ini = rng.normal(loc=0.0, scale=0.01, size=(X.shape[0], X.shape[0]))

        # Only coefficients corresponding to nearest neighbors should be non-zero
        mask = np.ones_like(W_ini, dtype=bool)
        for i, neighbors in nearestNeighborCollection.items():
            mask[i, neighbors] = False
        W_ini[mask] = 0

        self.similarity_matrix_, self.iters = self._bnslim_train(X.toarray(), p, W_ini, nearestNeighborCollection)

    def fit(self, X: Union[csr_matrix, np.ndarray], inner_ids_npr: list, W_ini: Optional[np.ndarray] = None):
        """
        Override the `fit` method so as to accept the ids of the non-protected group.
        """
        X = self._transform_fit_input(X) if not isinstance(X, csr_matrix) else X
        X = X if self.method == "user" else X.T

        p = np.ones(X.shape[0]); p[inner_ids_npr] = -1

        self._fit(X, p, W_ini)
        return self

    def _bnslim_train(self, X, p, W, nearestNeighborCollection):
        """
        Computes the coefficient matrix of BNSLIM.
        -------
        Args:
            X (np.ndarray): A user-item interaction matrix to fit the model to.
            p (np.ndarray): A vector indicating the group membership of each user/item.
                The membership is either 1 or -1, 1 for the users/items belonging to the protected group and -1 for the users/items belonging to the non-protected group.
            W (np.ndarray): The initial coefficient matrix.
            nearestNeighborCollection (dict): A dictionary storing the nearest neighbors for each item. 
                The keys of the dictionary should be item ids and the values should be lists of ids of nearest neighbors for each item. 
        -------
        Returns:
            W (np.ndarray): The updated coefficient matrix.
            iters (int): Number of iterations.
        """
        error = 10 * self.thr; iters = 0

        while (error > self.thr) and (iters < self.maxIter):

            W_0 = W.copy()

            for i in range(W.shape[0]):
                for k in nearestNeighborCollection[i]:
                    mask = nearestNeighborCollection[i].copy(); mask.remove(k)
                    X_ik = np.sum(X[i,:] - W[i,mask] @ X[mask,:]) + self.l3 * p[k] * np.sum(p[mask] * W[i,mask])
                    l2_norm_sq = np.linalg.norm(X[k,:])**2
                    update = np.multiply(np.sign(X_ik),np.maximum(np.abs(X_ik) - self.l1, 0)) / (l2_norm_sq + self.l2 + self.l3)
                    W[i, k] = 0 if update < 0 else update # original paper's solution - not in the librec's code

            error = np.max(np.abs(W_0 - W)); iters += 1

        return W, iters
    
    def _predict(self, X: csr_matrix) -> csr_matrix:
        """
        Override the `_predict` method so as to work for user-user similarities.
        """

        # Compute scores
        scores = self.similarity_matrix_ @ X if self.method == "user" else X @ self.similarity_matrix_.T

        # Convert to csr_matrix if not already one
        scores = csr_matrix(scores) if not isinstance(scores, csr_matrix) else scores

        return scores
    
    def _check_fit_complete(self):
        """
        Override the ` _check_fit_complete` method so as to work for user-user similarities.
        """
        # Use class to check is fitted
        Algorithm._check_fit_complete(self)

        # Additional checks on the fitted matrix
        # Check if actually exists!
        assert hasattr(self, "similarity_matrix_")

        # Check row wise, since that will determine the recommendation options
        items_with_score = set(self.similarity_matrix_.nonzero()[0])

        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            warnings.warn(f"{self.name} misses similarities for {missing} {self.method}s.")
    
    def get_recommendations(self, feedback, n=10):
        """
        Returns the top-n recommendations for a given user vector (inner ids).
        """

        # Estimate the ratings of user u
        estimates = np.dot(feedback, self.similarity_matrix_)

        # Exclude training examples
        estimates[feedback != 0] = -np.inf

        # Get the indices of the top-n items
        idx_topn = bn.argpartition(-estimates, n)[:n]

        # Sort the indices by the corresponding values in descending order
        idx_topn = sorted(idx_topn, key=lambda x: estimates[x], reverse=True)

        return idx_topn

    def get_neighbors(self, target, n=10):
        """
        Returns the strongest-n neighbors for a target user or item (inner ids).
        """
        
        coefficients = self.similarity_matrix_[:,target]

        # Get the indices of the n largest coefficients
        idx_strongest = bn.argpartition(-coefficients, n)[:n]

        # Sort the indices by the corresponding values in descending order
        idx_strongest = sorted(idx_strongest, key=lambda x: coefficients[x], reverse=True)

        return idx_strongest