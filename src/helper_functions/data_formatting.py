
import pandas as pd
import numpy as np
import scipy.sparse as sp
import bottleneck as bn

def get_sparse_matrix(data, user_col, item_col, rating_col):
    """
    Creates a sparse user-item interaction matrix from a given pandas DataFrame.
    -------
    Args:
        data (pandas.DataFrame): Dataset capturing user-item interactions.
        user_col (str): Column name representing user identifiers.
        item_col (str): Column name representing item identifiers.
        rating_col (str): Column name denoting the rating or interaction strength.
    -------
    Returns:
        A sparse matrix in Compressed Sparse Row (CSR) format representing the user feedback matrix.
    """
    row_indices = data[user_col].cat.codes
    col_indices = data[item_col].cat.codes
    ratings = data[rating_col]

    R = sp.csr_matrix((ratings, (row_indices, col_indices)), dtype=np.float64)

    return R

def get_raw_item_ids(df_pp, item_ids_inner):
    """
    Convert inner item IDs to their original raw form using the DataFramePreprocessor's item_id_mapping.
    -------
    Args:
        df_pp (recpack.preprocessing.preprocessors.DataFramePreprocessor): A DataFramePreprocessor object from the RecPack library.
        item_ids_inner (numpy.array): A NumPy array containing the inner item IDs.
    -------
    Returns:
        A NumPy array of raw item IDs.
    """
    item_id_mapping_df = df_pp.item_id_mapping
    iid_raw = item_id_mapping_df.columns[0]

    item_ids_raw = item_id_mapping_df[item_id_mapping_df["iid"].isin(item_ids_inner)][iid_raw].values

    return item_ids_raw

def get_raw_user_ids(df_pp, user_ids_inner):
    """
    Convert inner user IDs to their original raw form using the DataFramePreprocessor's user_id_mapping.
    -------
    Args:
        df_pp (recpack.preprocessing.preprocessors.DataFramePreprocessor): A DataFramePreprocessor object from the RecPack library.
        user_ids_inner (numpy.array): A NumPy array containing the inner user IDs.
    -------
    Returns:
        A NumPy array of raw user IDs.
    """
    user_id_mapping_df = df_pp.user_id_mapping
    uid_raw = user_id_mapping_df.columns[0]

    user_ids_raw = user_id_mapping_df[user_id_mapping_df["uid"].isin(user_ids_inner)][uid_raw].values

    return user_ids_raw

def get_inner_item_ids(df_pp, item_ids_raw):
    """
    Convert raw item IDs to their inner form using the DataFramePreprocessor's item_id_mapping.
    -------
    Args:
        df_pp (recpack.preprocessing.preprocessors.DataFramePreprocessor): A DataFramePreprocessor object from the RecPack library.
        item_ids_raw (numpy.array): A NumPy array containing the raw item IDs.
    -------
    Returns:
        A NumPy array of inner item IDs.
    """
    item_id_mapping_df = df_pp.item_id_mapping
    iid_raw = item_id_mapping_df.columns[0]

    item_ids_inner = item_id_mapping_df[item_id_mapping_df[iid_raw].isin(item_ids_raw)]["iid"].values

    return item_ids_inner

def get_inner_user_ids(df_pp, user_ids_raw):
    """
    Convert raw user IDs to their inner form using the DataFramePreprocessor's user_id_mapping.
    -------
    Args:
        df_pp (recpack.preprocessing.preprocessors.DataFramePreprocessor): A DataFramePreprocessor object from the RecPack library.
        user_ids_raw (numpy.array): A NumPy array containing the raw user IDs.
    -------
    Returns:
        A NumPy array of inner user IDs.
    """
    user_id_mapping_df = df_pp.user_id_mapping
    uid_raw = user_id_mapping_df.columns[0]

    user_ids_inner = user_id_mapping_df[user_id_mapping_df[uid_raw].isin(user_ids_raw)]["uid"].values

    return user_ids_inner

def get_topn_indices(R_hat, n):
    """
    Helper function to get sorted indices of top-n items in each row of R_hat.
    """
    users = R_hat.shape[0]
    
    # find the indices that partition the array so that the first n elements are the largest n elements
    idx_topn_part = bn.argpartition(-R_hat, n, axis=1)

    # keep only the largest n elements of R_hat
    topn_part = R_hat[np.arange(users)[:, np.newaxis], idx_topn_part[:, :n]]

    # find the indeces of the sorted top-n predicted relevance scores in R_hat
    idx_part = np.argsort(-topn_part, axis=1)
    idx_topn = idx_topn_part[np.arange(users)[:, np.newaxis], idx_part]
    
    return idx_topn
