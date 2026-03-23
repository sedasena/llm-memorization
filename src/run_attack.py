import argparse
import json
import math
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deepseek_r1_client import DeepSeekClient
from models.gemma_client import GemmaClient
#from models.llama3_1_8b_client import LlamaClient
from models.qwen3_1_7b_client import QwenClient

from src.generate_sequences import generate_sample, generate_batch
from evaluation_metrics.sliding_window import sliding_window_perplexity
from evaluation_metrics.zlib_ratio import zlib_ratio
from evaluation_metrics.compare_lm import compare_to_smaller_lm, cross_model_membership_score
from src.checkpoint import load_checkpoint, save_checkpoint
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
        
    #generator_lm = QwenClient("Qwen/Qwen3-1.7B") hier eigentlich llama als big model
    #reference_lms = {"small_1": GemmaClient("google/gemma-3-4b-it"),
                     #"small_2": DeepSeekClient("deepseek-ai/DeepSeek-R1")}
       # "qwen": QwenClient("Qwen/Qwen3-1.7B"), 
        #"gemma": GemmaClient("google/gemma-3-4b-it"),
        #"deepseek": DeepSeekClient("deepseek-ai/DeepSeek-R1")
    
        
    #results = []
    batch_size = 10
    num_batches = args.num_samples // batch_size
        # Checkpoint laden falls vorhanden
    results = load_checkpoint(args.model)
    start_batch = len(results) // batch_size
    print(f"Starting at batch {start_batch}/{num_batches}")
    
    for i in tqdm(range(num_batches)):
        texts = generate_batch(generator_lm, batch_size=batch_size)
        for text in texts:
            sw_gen = sliding_window_perplexity(generator_lm, text, 50, 10)
            zlib_score = zlib_ratio(generator_lm.perplexity(text), text)
            perplexity_gen = generator_lm.perplexity(text)
            perplexity_lower = generator_lm.perplexity(text.lower())
        
            lowercase_ratio = math.log(perplexity_lower + 1e-8) - math.log(perplexity_gen + 1e-8)
            # membership score compares both gen and ref lms perplexity
            score = cross_model_membership_score(generator_lm, reference_lms, text)
        
            results.append({
                "text": text,
                "sw_perplexity_gen": sw_gen,
                "zlib_ratio": zlib_score,
                "perplexity_gen": score["perplexity_gen"],                
                "perplexity_reference": score["perplexity_ref"],
                "lowercase_ratio": lowercase_ratio,
                "mean_ref": score["membership_score"],
                "ratio": score["mean_ref"] / (score["perplexity_gen"] + 1e-8),
                **score
            })
            
        if i % 100 == 0:
            save_checkpoint(results, args.model)
    
    save_checkpoint(results, args.model)
    # deduplicate elements in the list
    results = deduplicate(results)
    

    # Rank results
    rankings = {
        "perplexity": sorted(results, key=lambda x: x["perplexity_gen"]),
        "membership_score": sorted(results, key=lambda x: -x["membership_score"]),
        "sw_perplexity_gen": sorted(results, key=lambda x: x["sw_perplexity_gen"]),
        "ratio": sorted(results, key=lambda x: -x["ratio"]),
        "zlib_ratio": sorted(results, key=lambda x: -x["zlib_ratio"]),
        "lowercase_ratio": sorted(results, key=lambda x: -x["lowercase_ratio"]),
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
   