def sparse_dot_product(vec1, vec2):
    return sum([vec1[i] * vec2[i] for i in range(len(vec1))])