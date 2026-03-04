import time


def measure_latency(start, end):
    return end - start


def tokens_per_second(tokens, latency):
    if latency == 0:
        return 0
    return tokens / latency


def compute_metrics(tokens, start, end):
    latency = measure_latency(start, end)

    return {
        "latency": latency,
        "tokens_per_second": tokens_per_second(tokens, latency)
    }