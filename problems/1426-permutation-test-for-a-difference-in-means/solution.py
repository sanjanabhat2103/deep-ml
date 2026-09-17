import numpy as np


def difference_in_means(x, y):
  return np.mean(x) - np.mean(y)


def permutation_null(x, y, n_perm, seed=0):
  x = np.asarray(x)
  y = np.asarray(y)
  pooled = np.concatenate([x, y])
  nx = len(x)
  rng = np.random.default_rng(seed)
  diffs = np.empty(n_perm)
  for i in range(n_perm):
    permuted = rng.permutation(pooled)
    x_perm = permuted[:nx]
    y_perm = permuted[nx:]
    diffs[i] = np.mean(x_perm) - np.mean(y_perm)
  return diffs


def permutation_pvalue(x, y, n_perm, seed=0):
  observed = difference_in_means(x, y)
  null = permutation_null(x, y, n_perm, seed)
  pvalue = np.mean(np.abs(null) >= np.abs(observed))
  return round(float(pvalue), 4)


def permutation_test(x, y, n_perm, alpha, seed=0):
  observed = difference_in_means(x, y)
  pvalue = permutation_pvalue(x, y, n_perm, seed)
  reject = pvalue < alpha
  return observed, pvalue, reject