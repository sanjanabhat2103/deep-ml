def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    if not vectors:
        return []
    n_features = len(vectors)
    n_samples = len(vectors[0])
    if n_samples < 2:
        raise ValueError("At least two observations are required.")
    if any(len(v) != n_samples for v in vectors):
        raise ValueError("All vectors must have the same number of observations.")
    means = [sum(v) / n_samples for v in vectors]
    covariance = []
    for i in range(n_features):
        row = []
        for j in range(n_features):
            cov = sum(
                (vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
                for k in range(n_samples)
            ) / (n_samples - 1)
            row.append(float(cov))
        covariance.append(row)
    return covariance