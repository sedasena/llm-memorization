import random, math
import torch

def decay_temperature(step, max_decay_steps=20, start_temp=10.0, end_temp=1.0):
    """ Ensures decaying temperature. Temperature starts at 10 and decays down to 1 over the period
    of the first 20 tokens, which is approximately 10 percent of the length of the sequences."""
    if step >= max_decay_steps:
        return end_temp
    
    decay = step / max_decay_steps
    # lineare interpolation oder division?
    return start_temp + decay * (end_temp - start_temp)

def load_internet_prefixes(wet_file, num_prefixes=10000):
    prefixes = []
    with open(wet_file, 'r', errors='ignore') as f:
        text = f.read()
    
    words = text.split()
    for _ in range(num_prefixes):
        # Zufällige Position
        idx = random.randint(0, len(words) - 10)
        # 5-10 zufällige Wörter als Prefix
        length = random.randint(5, 10)
        prefix = " ".join(words[idx:idx+length])
        prefixes.append(prefix)
    
    return prefixes

def generate_sample(client, prefix="", total_tokens=256, top_k=40):
    """ Generates samples based on the chosen client, max token as well as top-k value
    analogous to Carlini et al. 2021. """
    tokenizer = client.tokenizer
    tokenizer.padding_side = "left"
    model = client.model
    device = client.device
    
    if prefix is None:
        input_ids = torch.tensor(
            [[tokenizer.eos_token_id]], device=device
        )
    else:
        input_ids = tokenizer(prefix, return_tensors="pt")["input_ids"].to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=total_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
        
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #return generated_text

def generate_batch(client, batch_size=10, total_tokens=256, top_k=40):
    """ Creates prompts and calls client.generate_batch """
    tokenizer = client.tokenizer
    tokenizer.padding_side = "left"
    model = client.model
    device = client.device
    
    prompts = [tokenizer.eos_token] * batch_size
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True
    ).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=total_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    results = []
    for ids in output_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        results.append(text)
    return results


def generate_batch_internet(client, prefixes, batch_size=10, total_tokens=256, top_k=40):
    """ Generate batches based on randomly sampled prefixes """
    tokenizer = client.tokenizer
    tokenizer.padding_side = "left"
    model = client.model
    device = client.device
    batch_prefixes = random.sample(prefixes, min(batch_size, len(prefixes)))
    
    inputs = tokenizer(
        batch_prefixes,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=10  
    ).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=total_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    results = []
    for ids in output_ids:
        results.append(tokenizer.decode(ids, skip_special_tokens=True))
    return results