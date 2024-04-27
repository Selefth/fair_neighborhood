
import warnings
import numpy as np
import bottleneck as bn

from scipy.sparse import csr_matrix

from recpack.algorithms.base import ItemSimilarityMatrixAlgorithm, Algorithm

class myEASE(ItemSimilarityMatrixAlgorithm):
    """
    Implementation of EASE (Embarrassingly Shallow AutoEncoders) recommender (Steck, 2019).
    Switch between i-i and u-u similarities by specifying the `method` parameter in the constructor.
    ----------
    Reference:
    Steck, Harald. Embarrassingly shallow autoencoders for sparse data. WWW 2019
    """

    def __init__(self, l2=1e2, method="item"):
        """
        Initializes parameters for EASE.
        """
        super().__init__()
        self.l2 = l2 
        self.method = method

    def _fit(self, X: csr_matrix):
        """
        Computes the coefficient matrix of EASE.
        """
        # Transpose the matrix for user-based approach
        if self.method == "user": X = X.T

        # Compute the P matrix
        P = (X.T @ X).toarray().astype("float32")
        dIndices = np.diag_indices(X.shape[1])
        P[dIndices] += self.l2

        # Compute the coefficient matrix W
        P = np.linalg.inv(P)
        # W = P / (-np.diag(P))
        W = P / (-np.einsum('ii->i', P)) # more efficient
        W[dIndices] = 0
        self.similarity_matrix_ = W

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """
        Override the `_predict` method so as to work for user-user similarities.
        """

        # Compute scores
        scores = self.similarity_matrix_.T @ X if self.method == "user" else X @ self.similarity_matrix_

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

        # Check column wise, since that will determine the recommendation options
        # TODO: Inform recpack authors; they do row wise check
        items_with_score = set(self.similarity_matrix_.nonzero()[1])

        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            warnings.warn(f"{self.name} misses similarities for {missing} {self.method}s.")
    
    def get_recommendations(self, feedback, n=10):
        """
        Returns the top-n recommendations for a given user vector (inner ids).
        """
        
        # feedback = self.X[user].toarray()[0]

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