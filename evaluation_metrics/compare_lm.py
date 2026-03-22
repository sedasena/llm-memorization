import math

def compare_to_smaller_lm(large_lm, small_lm, text):
    perplexity_large = large_lm.perplexity(text)
    perplexity_small = small_lm.perplexity(text)
    
    # ratio
    if perplexity_large > 0:
        ratio = perplexity_small / perplexity_large
    else:
        ratio = 0.0
        
    log_diff = math.log(perplexity_small + 1e-8) - math.log(perplexity_large + 1e-8)
    
    return {
        "perplexity_large": perplexity_large,
        "perplexity_small": perplexity_small,
        "ratio": ratio,
        "difference": log_diff
    }
    
def cross_model_membership_score(generator, reference_lms, text):
    """This function computes a cross-model membership inference score by aggregating by mean. """
    perplexity_gen = generator.perplexity(text)

    perplexity_ref = {
        name: lm.perplexity(text)
        for name, lm in reference_lms.items()
    }
    
    mean_ref = sum(perplexity_ref.values()) / len(perplexity_ref)
    score = math.log(mean_ref) - math.log(perplexity_gen)

    return {
        "perplexity_gen": perplexity_gen,
        "perplexity_ref": perplexity_ref,
        "membership_score": score
    }