def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.

    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check

    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    total_x = 0
    count_xy = 0
    for X, Y in data:
        if X == x:
            total_x += 1
            if Y == y:
                count_xy += 1
    if total_x == 0:
        return 0.0
    return round(count_xy / total_x, 4)
