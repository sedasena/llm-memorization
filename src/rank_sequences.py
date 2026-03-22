import math
from evaluation_metrics.zlib_ratio import zlib_ratio, zlib_entropy
from evaluation_metrics.sliding_window import sliding_window_perplexity

def rank_perplexity(samples):
    return sorted(samples, key=lambda x: x["perplexity_gen"])

def rank_by_small_lm(samples):
    return sorted(samples, key=lambda x: -x["membership_score"])

def rank_by_zlib(samples):
    return sorted(samples, key=lambda x: x["sw_perplexity_gen"])

def rank_by_lowercase(samples):
    return sorted(samples, key= lambda x: -x.get("lo"))