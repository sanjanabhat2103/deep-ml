def calculate_batch_health(predictions: list, confidence_threshold: float = 0.5) -> dict:
    """
    Calculate health metrics for a batch prediction job.
    
    Args:
        predictions: list of prediction results, each a dict with 'status' and optionally 'confidence'
        confidence_threshold: threshold below which a prediction is considered low confidence
    
    Returns:
        dict with keys: 'success_rate', 'avg_confidence', 'low_confidence_rate'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if len(predictions) == 0:
        return {}
    n = len(predictions)
    success = 0
    con = 0
    low_con = 0
    for i in predictions:
        if i['status'] == 'success':
            success += 1
            con += i['confidence']
            if i['confidence'] < confidence_threshold:
                low_con += 1
    success_rate = success / n * 100
    avg_confidence = con / success * 100 if success != 0 else 0.0
    low_confidence_rate = low_con / success * 100 if success != 0 else 0.0
    return {'success_rate': round(success_rate, 2), 'avg_confidence': round(avg_confidence, 2), 'low_confidence_rate': round(low_confidence_rate, 2)}