# Implement your function below.

def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1.
    """
    ref = reference.split()
    can = candidate.split()
    overlaps = []
    for i in ref:
        if i in can:
            overlaps.append(i)
    precision = len(overlaps) / len(can)
    recall = len(overlaps) / len(ref)
    f1 = 2 * precision * recall / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}