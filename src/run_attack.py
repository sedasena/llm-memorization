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

from src.generate_sequences import generate_sample, generate_batch, generate_batch_internet
from evaluation_metrics.sliding_window import sliding_window_perplexity
from evaluation_metrics.zlib_ratio import zlib_ratio
from evaluation_metrics.compare_lm import compare_to_smaller_lm, cross_model_membership_score
from src.checkpoint import load_checkpoint, save_checkpoint
import torch 
import sbert_verifier as SB

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

USE_INTERNET = True  # sample from internet
internet_prefixes = None

def deduplicate(samples):
    seen = set()
    unique = []
    
    for s in samples:
        if s["text"] not in seen:
            seen.add(s["text"])
            unique.append(s)
    return unique

def parse_common_crawl(wet_file_path, num_prefixes=10000):
    with open(wet_file_path) as f:
        for line in f:
            line.read
        lines = f.readlines()
        
    start_idx = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]
    all_eng = ""
    count_eng = 0
    for i in range(len(start_idx)-1):
        start = start_idx[i]
        end = start_idx[i+1]
        if "WARC-Identified-Content-Language: eng" in lines[start+7]:
            count_eng += 1
            for j in range(start+10, end):
                all_eng += lines[j]
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek", 
                choices=["qwen", "gemma", "deepseek", "llama"])
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    # Cross-model comparison with reference lms and llama3 8b as generator
    # generator_lm = LlamaClient("meta-llama/Llama-3-8B")
    if args.model == "deepseek":
        generator_lm = DeepSeekClient("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
        reference_lms = {"qwen_small": GemmaClient("google/gemma-3-4b-it"),
                          "deepseek_small": QwenClient("Qwen/Qwen3-1.7B")}
  
    #results = []
    batch_size = 10
    num_batches = args.num_samples // batch_size
    # laod checkpoint    
    results = load_checkpoint(args.model)
    start_batch = len(results) // batch_size
    
    print(f"Starting at batch {start_batch}/{num_batches}")
    
    if USE_INTERNET:
        print("Loading prefixes from the web...")
        cc_prefixes = parse_common_crawl(
            "CC-MAIN-20250126135402-20250126165402-00000.warc.wet.gz"
        )
        print(f"{len(cc_prefixes)} prefixes loaded.")
        
    for i in tqdm(range(num_batches)):
        if USE_INTERNET and cc_prefixes: # random internet sampling
            texts = generate_batch_internet(generator_lm, cc_prefixes, batch_size=batch_size)
        else:
            texts = generate_batch(generator_lm, batch_size=batch_size)
            
        for text in texts:
            perplexity_gen = generator_lm.calculate_perplexity(text)

            sw_gen = sliding_window_perplexity(generator_lm, text, 50, 10)
            zlib_score = zlib_ratio(perplexity_gen, text)
            perplexity_lower = generator_lm.calculate_perplexity(text.lower())
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
            
        if i % 100 == 0: # checkpointing for colab
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
   