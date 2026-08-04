import math

def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exps = [math.exp(x - max_score) for x in scores]
    total = sum(exps)
    return [x / total for x in exps]