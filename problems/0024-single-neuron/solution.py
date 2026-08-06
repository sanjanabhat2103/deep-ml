import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))
    probabilities = []
    for sample in features:
        z = sum(feature * weight for feature, weight in zip(sample, weights)) + bias
        probabilities.append(sigmoid(z))
    mse = sum((prob - label) ** 2 for prob, label in zip(probabilities, labels)) / len(labels)
    return probabilities, mse