### zlib-ratio ###

import zlib

def zlib_entropy(text):
    """ Compresses text sequence into the zlib compression format. """
    compressed = zlib.compress(text.encode("utf-8"))
    return len(compressed)

def zlib_ratio(perplexity, text):
    """ Computes the ratio of perplexity and zlib entropy as the zlib ratio (the membership inference metric). """
    entropy = zlib_entropy(text)
    
    if entropy == 0:
        return 0.0
    
    return perplexity / entropy

