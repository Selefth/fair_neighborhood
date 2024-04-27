
import numpy as np

from src.helper_functions.data_formatting import *

def coverage(R_hat, item_ids, n):
    """
    Coverage quantifies the proportion of items that are recommended out of the specified item set. 
    It is a metric indicative of the system's ability to diversify its recommendations. 
    A score close to 1 denotes higher diversity, suggesting that the system effectively recommends a vast array of items to users.
    -------
    Args:
        R_hat (numpy.ndarray): The estimated user feedback matrix with -np.inf marking entries used as model input.
        item_ids (lst): A list of item IDs for which to compute the coverage.
        n (int): The number of items to consider in the ranking.
    -------
    Returns:
        coverage (flt): The total coverage score.
    """

    # find the indices of the sorted top-n predicted relevance scores in R_hat
    idx_topn = get_topn_indices(R_hat, n)
    flat_topn = idx_topn.flatten()

    item_counts = {item_id: np.count_nonzero(flat_topn == item_id) for item_id in item_ids}  # count frequency of each item ID
    item_counts = np.array(list(item_counts.values()))
    coverage = sum(np.minimum(1, item_counts)) / len(item_ids)

    return coverage

def u_parity_at_n(R_hat, protected_users, items_dict, n):
    """
    Computes the u-parity@N score. 
    The User-coverage Parity metric measures the average disparity between the proportions of protected 
    and non-protected users receiving recommendations from each category.
    -------
    Args:
    R_hat (numpy.ndarray): The estimated user feedback matrix with -np.inf marking entries used as model input.
    protected_users (numpy.ndarray): A binary array where 1 indicates a user is part of the protected group and 0 otherwise.
    items_dict (dict): A dictionary mapping item categories to their corresponding item IDs.
    n (int): The number of items to consider in the ranking.
    -------
    Returns:
    u_parity (flt): The u-parity@N score.
    std_dev (flt): The standard deviation of the differences across item categories.
    """

    # get top-n indices for all users
    idx_topn = get_topn_indices(R_hat, n)

    # compute the number of protected and non-protected users once
    num_protected_users = sum(protected_users)
    num_non_protected_users = len(protected_users) - num_protected_users

    differences = []
    for _, item_ids in items_dict.items():
        count_protected_users = 0; count_non_protected_users = 0
        for i, is_protected in enumerate(protected_users):
            if np.isin(idx_topn[i], item_ids).sum() > 0:
                if is_protected:
                    count_protected_users += 1 
                else:
                    count_non_protected_users += 1

        # normalize the number of recommendations by the total number of users in each group
        prop_protected_users = count_protected_users / num_protected_users
        prop_non_protected_users = count_non_protected_users / num_non_protected_users

        differences.append(abs(prop_non_protected_users - prop_protected_users))

    u_parity = np.mean(differences); std_dev = np.std(differences)

    return u_parity, std_dev

def apcr_at_n(R_hat, items_dict, n):
    """
    Computes the apcr@N score.
    This metric measures the average rate at which providers are covered in the recommendation lists for a set of users.
    -------
    Args:
    R_hat (numpy.ndarray): The estimated user feedback matrix with -np.inf marking entries used as model input.
    items_dict (dict): A dictionary mapping providers to their corresponding item IDs.
    n (int): The number of items to consider in the ranking.
    -------
    Returns:
    apcr (flt): The apcr@N score.
    -------
    Reference:
        Weiwen Liu and Robinn Burke
        Personalizing Fairness-aware Re-ranking. FATREC Workshop on Responsible Recommendation 2018
    """

    # reverse the items_dict to map item IDs to provider IDs
    providers_dict = {item: provider for provider, items in items_dict.items() for item in items}

    idx_topn = get_topn_indices(R_hat, n)

    count_providers = []
    for user_row in idx_topn:
        user_providers = set(providers_dict.get(item) for item in user_row)
        count_providers.append(len(user_providers))

    apcr = np.mean(count_providers) / len(items_dict)

    return apcr