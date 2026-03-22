### Sliding Window Perplexity ###

import torch

def sliding_window_perplexity(client, text, window_size=50, stride=10): # check stride later in the paper
    """ This function calculates the minimum sliding perplexity according to a window size of 50 tokens. """
    tokenizer = client.tokenizer
    model = client.model
    device = client.device
    
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
    
    if len(tokens) < window_size:
        return client.perplexity(text)
    
    min_perplexity = float("inf")
    
    for tok in range(0, len(tokens) - window_size + 1, stride):
        window_tokens = tokens[tok: tok + window_size]
        
        inputs = {
            "input_ids": window_tokens.unsqueeze(0).to(device),  # increase tensor dimension by one
            "labels": window_tokens.unsqueeze(0).to(device)
            }
        
        with torch.no_grad():
            outputs = model(**inputs)
            perplexity = torch.exp(outputs.loss).item()
        
        # select the minimum of perplexity
        min_perplexity = min(min_perplexity, perplexity)
        
    return min_perplexity
