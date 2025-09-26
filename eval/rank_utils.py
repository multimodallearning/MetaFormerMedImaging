# based on https://github.com/MDL-UzL/L2R/blob/main/ranking/ranking_utils.py
import itertools

import torch
from scipy.stats import ranksums


def scores_better(tasks_metric: torch.Tensor, test_method:str, alpha: float = 0.05):
    """

    :param tasks_metric: token_mixer x n_test_scores
    :param test_method: str Method used to calculate significance
    :param alpha: significance level
    :return:
    """
    T, S = tasks_metric.shape
    better = torch.full((T, T), -1)
    for t_curr, t_comp in itertools.product(range(T), repeat=2):
        if test_method == "ranksums":
            h, p = ranksums(tasks_metric[t_curr].numpy(), tasks_metric[t_comp].numpy())
            if h > 0 and p < alpha:  # sign of h and p-value
                better[t_curr, t_comp] = 1
            else:
                better[t_curr, t_comp] = 0
        elif test_method == "CI-based":
            diffs = tasks_metric[t_curr] - tasks_metric[t_comp]
            ci_low = torch.quantile(diffs, alpha/2)
            ci_high = torch.quantile(diffs, 1-alpha/2)
            if ci_low > 0 or ci_high < 0: # significant
                better[t_curr, t_comp] = 1 if diffs.median() > 0 else 0 # and better
            else:
                better[t_curr, t_comp] = 0  # non significant
        else:
            raise ValueError(f"Unknown test_method: {test_method}. Please select from [ranksums, CI-based]")
    assert (better != -1).all(), 'some comparisons have not been made'
    scores_task = better.sum(1)
    return scores_task


def rankscore_avgtie(scores_int: torch.Tensor) -> torch.Tensor:
    """
    Compute averaged tie ranks scaled to [0.1, 1] for integer scores.

    :param scores_int: scores tensor as integer
    :return: averaged tie ranks
    """
    assert scores_int.dtype == torch.int64 and scores_int.dim() == 1
    N = scores_int.shape[0]

    # Rank scale: from 0.1 to 1
    rankscale = torch.linspace(0.1, 1.0, N)

    # argsort to assign rank positions
    order = torch.argsort(scores_int)
    ranks = torch.empty(N, dtype=torch.float)
    ranks[order] = rankscale

    # unique scores + inverse mapping
    unique_scores, inv = torch.unique(scores_int, return_inverse=True)

    # sums and counts by group
    rank_sum = torch.zeros_like(unique_scores, dtype=torch.float)
    rank_count = torch.zeros_like(unique_scores, dtype=torch.float)

    rank_sum.scatter_add_(0, inv, ranks)
    rank_count.scatter_add_(0, inv, torch.ones_like(ranks))

    # average rank per group
    avg_rank = rank_sum / rank_count.clamp_min(1e-6)

    # map back to original scores
    scorerank = avg_rank[inv]

    return scorerank


if __name__ == '__main__':
    scores = torch.tensor([2, 2, 3, 5, 3, 1])
    scores_normalized = rankscore_avgtie(scores)
    print(scores_normalized)
