import re

# trigram-multisets of a sample s
def get_trigrams(sample):
    """ Returns word-level trigrams, split on whitespace and punctuation marks"""
    tokens = re.split(r'[\s\.,!?;:()\[\]{}\"\'-]+', sample.lower())
    tokens = [token for token in tokens if token]
    
    result = []
    for i in range(len(tokens)-2):
        result = [tuple(tokens[i:i+3])]
    return result
    # return [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    
def is_duplicate(sample_1, sample_2):
    """ Mark sample_1 as duplicate of sample_2 if:
        |tri(sample_1) ∩ tri(sample_2) | |tri(sample_1)| / 2,
        as stated in the Data De-Duplication section in Carlini et al. 2021, 6.1."""
    tri1 = get_trigrams(sample_1)
    tri2 = get_trigrams(sample_2)
        
    if tri1 or tri2 == False:
        return True
    
    set2 = set(tri2)
    overlap = 0
    for t in tri2:
        if t in set2:
           overlap += 1 
           
    return overlap >= len(tri1) / 2
        
def deduplicate(samples, key="text"):
    """ Fuzzy deduplication with trigram-overlap. """
    unique = []
    is_dup = False
    for sample in samples:
        candidate_text = sample[key]
        
        for seen in unique:
            if is_duplicate(candidate_text, seen[key]):
                is_dup = True
                break
            if not is_dup:
                unique.append(sample)
    return unique