import argparse
import json
import math
from tqdm import tqdm

from models.deepseek_r1_client import DeepSeekClient
from models.gemma_client import GemmaClient
#from models.llama3_1_8b_client import LlamaClient
from models.qwen3_1_7b_client import QwenClient

from src.generate_sequences import generate_sample
from evaluation_metrics.sliding_window import sliding_window_perplexity
from evaluation_metrics.zlib_ratio import zlib_ratio
from evaluation_metrics.compare_lm import compare_to_smaller_lm, cross_model_membership_score

import torch 
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
def deduplicate(samples):
    seen = set()
    unique = []
    
    for s in samples:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", 
                choices=["qwen", "gemma", "deepseek", "llama"])
    parser.add_argument("--num_samples", type=int, default=200000)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    
    
    # Cross-model comparison with reference lms and llama3 8b as generator
    #generator_lm = LlamaClient("meta-llama/Llama-3-8B")
    
    if args.model == "qwen":
        generator_lm = QwenClient("Qwen/Qwen3-1.7B")
        reference_lms = {"small_1": GemmaClient("google/gemma-3-4b-it")}
                    # "small_2": DeepSeekClient("deepseek-ai/DeepSeek-R1")
   # elif args.model == "gemma":
   #     generator_lm = GemmaClient("google/gemma-3-4b-it")
   #     reference_lms = {"small_1": GemmaClient("google/gemma-3-4b-it"),
    #                 "small_2": DeepSeekClient("deepseek-ai/DeepSeek-R1")}
        
    #generator_lm = QwenClient("Qwen/Qwen3-1.7B")
    #reference_lms = {"small_1": GemmaClient("google/gemma-3-4b-it"),
                     #"small_2": DeepSeekClient("deepseek-ai/DeepSeek-R1")}
       # "qwen": QwenClient("Qwen/Qwen3-1.7B"), 
        #"gemma": GemmaClient("google/gemma-3-4b-it"),
        #"deepseek": DeepSeekClient("deepseek-ai/DeepSeek-R1")
    
        
    results = []

    for _ in tqdm(range(args.num_samples)):
        text = generate_sample(generator_lm) # generated samples are based on the larger lm
        sw_gen = sliding_window_perplexity(generator_lm, text, 50, 10)
        zlib_score = zlib_ratio(generator_lm.perplexity(text), text)
        perplexity_gen = generator_lm.perplexity(text)
        perplexity_lower = generator_lm.perplexity(text.lower())
        
        lowercase_ratio = math.log(perplexity_lower + 1e-8) - math.log(perplexity_gen + 1e-8)
        # membership score compares both gen and ref lms perplexity
        score = cross_model_membership_score(generator_lm, reference_lms, text)
        
        results.append({
            "text": text,
            "perplexity_ref": score["perplexity_ref"],
            "perplexity_gen": score["perplexity_gen"],
            "ref_mean": score["membership_score"],
            "sw_perplexity_gen": sw_gen,
            "zlib_ratio": zlib_score,
            **score
        })
    
    # deduplicate elements in the list
    results = deduplicate(results)
    
    # Rank results
    rankings = {
        "sw_perplexity_gen": sorted(results, key=lambda x: x["sw_perplexity_gen"]),
        "ratio": sorted(results, key=lambda x: -x["ratio"]),
        "zlib_ratio": sorted(results, key=lambda x: -x["zlib_ratio"]),
    }
    
    TOP_K = 100
    
    
    with open(args.output, "w") as f:
        json.dump(
            {"results": results,
             "top_k": {
                metric: ranked[:TOP_K] # search the top 100 manually
                for metric, ranked in rankings.items()
             }
                
            }, f, indent=2)
    
    print(f"Saved {len(results)} samples to {args.output}")
    

if __name__ == "__main__":
    main()