def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    """
    Implement a hard voting classifier using majority vote.
    
    Args:
        predictions: 2D list where predictions[i][j] is classifier i's prediction for sample j
        
    Returns:
        List of final predictions using majority vote
    """
    if not predictions:
        return []
    n_samples = len(predictions[0])
    final_predictions = []
    for j in range(n_samples):
        votes = [predictions[i][j] for i in range(len(predictions))]
        majority_vote = max(set(votes), key=votes.count)
        final_predictions.append(majority_vote)
    return final_predictions