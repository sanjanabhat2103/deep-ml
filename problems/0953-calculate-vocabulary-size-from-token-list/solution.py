def vocab_size(tokens, special_tokens=None):
    """Return the vocabulary size from a list of preprocessed tokens.

    Args:
        tokens: list[str] of preprocessed tokens
        special_tokens: optional list[str] of special tokens to include
    Returns:
        int: number of unique tokens in the combined vocabulary
    """
    vocabulary = set(tokens)
    if special_tokens:
        for i in special_tokens:
            if i not in vocabulary:
                vocabulary.add(i)
    return len(vocabulary)