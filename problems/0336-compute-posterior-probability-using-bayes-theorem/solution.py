def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
    """
    Calculate posterior probabilities using Bayes' Theorem.

    Args:
        priors: Prior probabilities P(H_i) for each hypothesis
        likelihoods: Likelihoods P(E|H_i) for each hypothesis

    Returns:
        Posterior probabilities P(H_i|E) for each hypothesis
    """
    if len(priors) != len(likelihoods):
        return [0.0, 0.0]
    evidence = sum(p * l for p, l in zip(priors, likelihoods))
    if evidence == 0:
        return [0.0] * len(priors)
    post_prob = []
    for i in range(len(priors)):
        post_prob.append((likelihoods[i] * priors[i]) / evidence)
    return post_prob
