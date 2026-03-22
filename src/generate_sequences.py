import random, math

def decay_temperature(step, max_decay_steps=20, start_temp=10.0, end_temp=1.0):
    """ This function ensures decaying temperature. Temperature starts at 10 and decays down to 1 over the period
    of the first 20 tokens, which is approximately 10 percent of the length of the sequences."""
    if step >= max_decay_steps:
        return end_temp
    
    decay = step / max_decay_steps
    # lineare interpolation oder division?
    return start_temp + decay * (end_temp - start_temp)


def generate_sample(client, prefix="", total_tokens=256, top_k=40):
    """ Generates samples based on the chosen client, max token as well as top-k value
    analogous to Carlini et al. 2021. """
    generated_text = prefix
    
    for token in range(total_tokens):
        temperature = decay_temperature(token)
        generated_text = client.generate(
            prompt=generated_text,
            max_new_tokens=1,
            temperature=temperature,
            top_k=top_k
        )
    return generated_text