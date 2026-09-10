from collections import Counter

def dummy_classifier(y_train, n_test, strategy, constant=None):
    counts = Counter(y_train)
    classes = sorted(counts)
    if strategy == "most_frequent":
        prediction = min(classes, key=lambda c: (-counts[c], c))
        return [prediction] * n_test
    elif strategy == "constant":
        return [constant] * n_test
    elif strategy == "uniform":
        return [classes[i % len(classes)] for i in range(n_test)]
    elif strategy == "stratified":
        n = len(y_train)
        allocations = {}
        fractions = {}
        for c in classes:
            exact = n_test * counts[c] / n
            allocations[c] = int(exact)
            fractions[c] = exact - allocations[c]
        remaining = n_test - sum(allocations.values())
        order = sorted(classes, key=lambda c: (-fractions[c], c))
        for c in order[: remaining]:
            allocations[c] += 1
        result = []
        for c in classes:
            result.extend([c] * allocations[c])

        return result
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
