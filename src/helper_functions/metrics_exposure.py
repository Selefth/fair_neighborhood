
import numpy as np

from src.helper_functions.data_formatting import *

def c_equity_original_at_n(R_hat_p, R_hat_np, items_dict, n):
    """
    Computes the original c-Equity@N score for each item category.
    A score less than 1 indicates that, on average, the protected group receives fewer recommendations of the respective category. 
    Conversely, a score greater than 1 suggests that the protected group is recommended more items of that category.
    -------
    Args:
        R_hat_p (numpy.ndarray): The estimated user feedback matrix for protected users with -np.inf marking entries used as model input.
        R_hat_np (numpy.ndarray): The estimated user feedback matrix for non-protected users with -np.inf marking entries used as model input.
        items_dict (dict): A dictionary with item categories as keys and item IDs as values.
        n (int): The number of items to consider in the ranking.
    -------
    Returns:
        equity (dict): The c-Equity@N score per category.
    -------
    Reference:
        Robin Burke, Nasim Sonboli, Aldo Ordonez-Gauger
        Balanced Neighborhoods for Multi-sided Fairness in Recommendation. FAccT 2018
    """

    users_p, users_np = R_hat_p.shape[0], R_hat_np.shape[0]

    # sorted top-n indices for protected and non-protected users
    idx_topn_sorted_p = get_topn_indices(R_hat_p, n)
    idx_topn_sorted_np = get_topn_indices(R_hat_np, n)

    equity = {}
    for category, item_ids in items_dict.items():
        numerator = sum(np.isin(idx_topn_sorted_p, item_ids).astype(int).sum(axis=1) / users_p) # how many items a user from the protected group consumed on average
        denominator = sum(np.isin(idx_topn_sorted_np, item_ids).astype(int).sum(axis=1) / users_np)
        # If the denominator is zero, we are in a situation where no items from the current category
        # are recommended to any user in the non-protected group among their top 'n' recommendations (rare case).
        if denominator == 0:
            equity[category] = np.inf
        else:
            equity[category] = numerator / denominator

    return equity

def c_equity_at_n(R_hat_p, R_hat_np, items_dict, n):
    """
    Computes the c-Equity@N score to evaluate equity across item categories. This function provides a concise summary by reporting 
    both the aggregate score and the variability in equity using standard deviation.
    -------
    Args:
        R_hat_p (numpy.ndarray): The estimated user feedback matrix for protected users with -np.inf marking entries used as model input.
        R_hat_np (numpy.ndarray): The estimated user feedback matrix for non-protected users with -np.inf marking entries used as model input.
        items_dict (dict): A dictionary with item categories as keys and item IDs as values.
        n (int): The number of items to consider in the ranking.
    -------
    Returns:
        c_equity (flt): The c-Equity@N score.
        std_dev (flt): The standard deviation of the differences across item categories.
        probabilities (dict): Dictionary with item categories as keys and a list of probabilities for protected and non-protected users as values.
    -------
    """

    users_p, users_np = R_hat_p.shape[0], R_hat_np.shape[0]

    # sorted top-n indices for protected and non-protected users
    idx_topn_sorted_p = get_topn_indices(R_hat_p, n)
    idx_topn_sorted_np = get_topn_indices(R_hat_np, n)

    differences = []; probabilities = {}
    for category, item_ids in items_dict.items():
        avg_recommended_p = sum(np.isin(idx_topn_sorted_p, item_ids).astype(int).sum(axis=1)) / (n*users_p)
        avg_recommended_np = sum(np.isin(idx_topn_sorted_np, item_ids).astype(int).sum(axis=1)) / (n*users_np)
        differences.append(abs(avg_recommended_p - avg_recommended_np))
        probabilities[category] = [avg_recommended_p, avg_recommended_np]

    c_equity = np.mean(differences); std_dev = np.std(differences)

    return c_equity, std_dev, probabilities

def p_equity_original_at_n(R_hat, R_held, items_dict, n):
    """
    Computes the original p-Equity@N score.
    The score will be less (more) than 1 when items from the protected group are appearing less (more) often on recommendation lists.
    -------
    Args:
        R_hat (numpy.ndarray): The estimated user feedback matrix with -np.inf marking entries used as model input.
        R_held (sparse matrix): The held out user feedback matrix.
        items_dict (dict): A dictionary with item categories as keys and item IDs as values.
        n (int): The number of items to consider in the ranking.
    -------
    Returns:
        equity (flt): The p-Equity@N score.
    -------
    Reference:
        Robin Burke, Nasim Sonboli, Aldo Ordonez-Gauger
        Balanced Neighborhoods for Multi-sided Fairness in Recommendation. FAccT 2018
    """

    if len(items_dict) != 2:
        raise ValueError("items_dict should have exactly two keys.")

    keys = list(items_dict.keys())
    protected_key = keys[0]; non_protected_key = keys[1]

    # find the indeces of the sorted top-n predicted relevance scores in R_hat
    idx_topn = get_topn_indices(R_hat, n)

    # number of un-interacted items from the (non-)protected group being recommended per user
    count_p = np.isin(idx_topn, items_dict[protected_key]).astype(int).sum(axis=1)
    count_np = np.isin(idx_topn, items_dict[non_protected_key]).astype(int).sum(axis=1)

    # R_held columns that have at least one non-zero element (items in testset)
    testset_item_ids = np.unique(R_held.indices)

    # number of (non-)protected items in the testset
    total_count_p = np.sum(np.isin(testset_item_ids, items_dict[protected_key]))
    total_count_np = np.sum(np.isin(testset_item_ids, items_dict[non_protected_key]))

    equity = sum(count_p/total_count_p)/sum(count_np/total_count_np)

    return equity

def bdv_at_n(R_hat, items_dict, n):
    """
    Bilateral Disparate Visibility (BDV).
    Computes the bdv@N score.
    -------
    Args:
        R_hat (numpy.ndarray): The estimated user feedback matrix with -np.inf marking entries used as model input.
        items_dict (dict): A dictionary with item categories as keys and item IDs as values.
        n (int): The number of items to consider in the ranking.
    -------
    Returns:
        bdv (flt): The bdv@N score.
    -------
    """

    if len(items_dict) != 2:
        raise ValueError("items_dict should have exactly two keys.")

    keys = list(items_dict.keys())
    protected_key = keys[0]; non_protected_key = keys[1]

    # find the indeces of the sorted top-n predicted relevance scores in R_hat
    idx_topn = get_topn_indices(R_hat, n)

    # number of un-interacted items from the (non-)protected group being recommended per user
    count_p = np.isin(idx_topn, items_dict[protected_key]).astype(int).sum(axis=1)
    count_np = np.isin(idx_topn, items_dict[non_protected_key]).astype(int).sum(axis=1)

    bdv = abs(sum(count_p)/(R_hat.shape[0]*n)-sum(count_np)/(R_hat.shape[0]*n))

    return bdv

def rsp_at_n(R_hat, R_held, items_dict, n):
    """
    Ranking-based Statistical Parity (RSP). Lower values indicate the recommendations are less biased.
    RSP-based bias leads to discrimination appearing in situations where people or items with sensitive information are recommended.
    -------
    Args:
        R_hat (numpy.ndarray): The estimated user feedback matrix with -np.inf marking entries used as model input.
        R_held (sparse matrix): The held in user feedback matrix.
        items_dict (dict): A dictionary with item categories as keys and item IDs as values.
        n (int): The number of items to consider in the ranking.
    -------
    Returns:
        rsp (flt): The rsp@N score.
    -------
    Reference:
        Ziwei Zhu, Jianling Wang, James Caverlee
        Measuring and Mitigating Item Under-Recommendation Bias in Personalized Ranking Systems. SIGIR 2020
    """

    users = R_hat.shape[0]

    # find the indices of the sorted top-n predicted relevance scores in R_hat
    idx_topn = get_topn_indices(R_hat, n)

    # ids of un-interacted items per user
    uninter_items = [np.where(R_held[i].toarray().flatten() == 0)[0] for i in range(users)]

    groups = []
    for group_ids in items_dict.values():
        # number of un-interacted items from the group being recommended per user
        count_rec_group = np.isin(idx_topn, group_ids).astype(int).sum(axis=1)

        count_held_group = np.zeros(users)
        for u, items in enumerate(uninter_items):
            # number of un-interacted items from the group per user
            count_held_group[u] = sum(np.isin(items, group_ids).astype(int))

        groups.append(sum(count_rec_group) / sum(count_held_group))

    # the relative standard deviation (to keep the same scale for different n over the probabilities)
    rsp = np.std(groups) / np.mean(groups)

    return rsp

def reo_at_n(R_hat, R_held, items_dict, n):
    """
    Ranking-based Equal Opportunity (REO). Lower values indicate the recommendations are less biased.
    REO evaluates the probability of being ranked in top-n given the ground-truth that the user likes the item.
    REO is supposed to be enhanced in general item recommendation systems so that no user need is ignored, and all items have the chance to be exposed to users who like them.
    REO considers the number of protected and non-protected items that are hidden per user.
    -------
    Args:
        R_hat (numpy.ndarray): The estimated user feedback matrix with -np.inf marking entries used as model input.
        R_held (sparse matrix): The held out user feedback matrix.
        items_dict (dict): A dictionary with item categories as keys and item IDs as values.
        n (int): The number of items to consider in the ranking.
    -------
    Returns:
        reo (flt): The reo@N score.
    -------
    Reference:
        Ziwei Zhu, Jianling Wang, James Caverlee
        Measuring and Mitigating Item Under-Recommendation Bias in Personalized Ranking Systems. SIGIR 2020
    """

    # find the indices of the sorted top-n predicted relevance scores in R_hat
    idx_topn = get_topn_indices(R_hat, n)

    # the hidden item ids per user (liked items)
    hidden_ids = [R_held[row].nonzero()[1] for row in range(R_held.shape[0])]

    groups = []
    for group_ids in items_dict.values():
        # find the recommended group items per user
        group_items_per_user = [np.intersect1d(row, group_ids) for row in idx_topn]

        # number of un-interacted items from the group being hidden but recommended per user
        count_rec_group = []
        for group_items, hids in zip(group_items_per_user, hidden_ids):
            count_rec_group.append(np.intersect1d(group_items, hids).size)

        # extract the rows of R_held corresponding to the group items
        R_group = R_held[:, group_ids]

        # count the total number of ones per row for the group items
        count_held_group = np.array(R_group.sum(axis=1)).flatten()

        groups.append(sum(count_rec_group) / sum(count_held_group))

    reo = np.std(groups) / np.mean(groups)

    return reo

def dp_at_n(R_hat_p, R_hat_np, n):
    """
    Compute Demographic Parity (DP@N) score. 
    The normalized absolute difference in the number of times each item appears in the top-N recommendations across the two groups.
    ----------
    Args:
        R_hat_p (numpy.ndarray): The estimated user feedback matrix for protected users with -np.inf marking entries used as model input.
        R_hat_np (numpy.ndarray): The estimated user feedback matrix for non-protected users with -np.inf marking entries used as model input.
        n (int): The number of items to consider in the ranking.
    -------
    Returns:
        dp (float): The DP@N score.
    -------
    Reference:
        Lei Chen, Le Wu, et al.
        Improving Recommendation Fairness via Data Augmentation. WWW 2023
    """

    idx_topn_sorted_p = get_topn_indices(R_hat_p, n)
    idx_topn_sorted_np = get_topn_indices(R_hat_np, n)
    
    # count how many times each item appears in the top-n lists
    item_counts_p = np.zeros(R_hat_p.shape[1])
    item_counts_np = np.zeros(R_hat_np.shape[1])

    for user_items in idx_topn_sorted_p: # for each user
        item_counts_p[user_items] += 1
    for user_items in idx_topn_sorted_np:
        item_counts_np[user_items] += 1

    normalization = np.maximum(item_counts_p + item_counts_np, 1)
    dp = (np.abs(item_counts_p - item_counts_np) / normalization).sum() / R_hat_np.shape[1]

    return dp