def calculate_inference_stats(latencies_ms: list) -> dict:
    """
    Calculate inference statistics for model monitoring.
    
    Args:
        latencies_ms: list of latency measurements in milliseconds
    
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """
    if not latencies_ms:
        return {}
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
    throughput_per_sec = 1000 / avg_latency_ms
    sorted_latencies = sorted(latencies_ms)
    def percentile(p):
        index = (len(latencies_ms) - 1) * p
        lower = int(index)
        upper = min(lower + 1, len(latencies_ms) - 1)
        fraction = index - lower
        return (sorted_latencies[lower] + fraction * (sorted_latencies[upper] - sorted_latencies[lower]))
    return {
        "throughput_per_sec": round(throughput_per_sec, 2), "avg_latency_ms": round(avg_latency_ms, 2), "p50_ms": round(percentile(0.50), 2), "p95_ms": round(percentile(0.95), 2), "p99_ms": round(percentile(0.99), 2),}