def build_vocab(tokens):
    """
    Build a vocabulary dictionary from a list of tokens.

    Args:
        tokens: list of string tokens

    Returns:
        Dict mapping each unique token (sorted) to a unique integer ID starting from 0.
    """
    unique = set(tokens)
    unique_sort = sorted(unique)
    d = {}
    for i in range(len(unique_sort)):
        d[unique_sort[i]] = i
    return d
