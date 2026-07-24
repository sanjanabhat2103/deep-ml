def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    means = []
    if not matrix or not matrix[0]:
        return means
    m = len(matrix)
    n = len(matrix[0])
    if mode == "column":
        for j in range(n):
            s = 0
            for i in range(m):
                s += matrix[i][j]
            means.append(s / m)
    elif mode == "row":
        for i in range(m):
            means.append(sum(matrix[i]) / n)
    return means