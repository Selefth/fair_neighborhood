
import numpy as np
import bottleneck as bn

from src.helper_functions.data_formatting import *

def recall_at_n(R_hat,R_held,n):
    """
    Computes the Recall at the given value of n.
    This metric does not take the rank of recommended items into account.
    Only for usage with binary feedback data.
    -------
    Args:
        R_hat (numpy.ndarray): The estimated user feedback matrix with -np.inf marking entries used as model input.
        R_held (sparse matrix): The held out user feedback matrix.
        n (int): The number of items to consider in the ranking.
    -------
    Returns:
        recall_mean (flt): The mean Recall@N across users.
        recall_std (flt): The standard deviation of Recall@N across users.
    """

    users = R_hat.shape[0]

    # find the indices that partition the array so that the first n elements are the largest n elements
    idx = bn.argpartition(-R_hat, n, axis=1)

    R_hat_binary = np.zeros_like(R_hat, dtype=bool)
    R_hat_binary[np.arange(users)[:, np.newaxis], idx[:, :n]] = True

    R_held_binary = (R_held > 0).toarray()

    # recall@N for each user
    recall = (np.logical_and(R_held_binary, R_hat_binary).sum(axis=1)).astype(np.float32) / np.minimum(n, R_held_binary.sum(axis=1))
    
    recall_mean = np.mean(recall); recall_std = np.std(recall)
    
    return recall_mean, recall_std

def tndcg_at_n(R_hat,R_held,n):
    """
    Computes the truncated Normalized Discounted Cumulative Gain (NDCG) at the given value of n.
    A score of 1 is achieved when dcg = idcg (ideal dcg).
    Only for usage with binary feedback data.
    -------
    Args:
        R_hat (numpy.ndarray): The estimated user feedback matrix with -np.inf marking entries used as model input.
        R_held (sparse matrix): The held out user feedback matrix.
        n (int): The number of items to consider in the ranking.
    -------
    Returns:
        tndcg_mean (flt): The mean truncated NDCG@N across users.
        tndcg_std (flt): The standard deviation of truncated NDCG@N across users.
    """

    users = R_hat.shape[0]

    # find the indeces of the sorted top-n predicted relevance scores in R_hat
    idx_topn = get_topn_indices(R_hat, n)

    tp = 1. / np.log2(np.arange(2, n + 2))
    dcg = (R_held[np.arange(users)[:, np.newaxis], idx_topn].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(i, n)]).sum() for i in R_held.getnnz(axis=1)])
    tndcg = dcg / idcg

    tndcg_mean = np.mean(tndcg); tndcg_std = np.std(tndcg)
    
    return tndcg_mean, tndcg_std
