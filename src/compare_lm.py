import math

def compare_to_smaller_lm(large_lm, small_lm, text):
    perplexity_large = large_lm.perplexity(text)
    perplexity_small = small_lm.perplexity(text)
    
    if perplexity_large > 0:
        ratio = perplexity_small / perplexity_large
    else:
        ratio = 0.0
        
    log_diff = math.log(perplexity_small + 1e-8) - math.log(perplexity_large + 1e+8)
    
    return {
        "perplexity_large": perplexity_large,
        "perplexity_small": perplexity_small,
        "ratio": ratio,
        "log_difference": log_diff
    }