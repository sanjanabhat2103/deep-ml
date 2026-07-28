def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    map = {}
    for i in samples:
        if i not in map:
            map[i] = 1
        else:
            map[i] += 1
    n = len(samples)
    pmf = []
    for i in map:
        pmf.append((i, map[i] / n))
    return sorted(pmf)