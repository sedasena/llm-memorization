import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sbert_verifier as SB

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)   # results_qwen.json
    parser.add_argument("--output", required=True)  # verified_qwen.json
    parser.add_argument("--metric", default="zlib_ratio")  # which metric for verification
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Top-K Kandidaten aus gewähltem Ranking
    candidates = data["top_k"][args.metric][:args.top_k]
    print(f"Verify {len(candidates)} candidates...")

    verifier = SB.SBertVerifier(threshold=0.85)
    verified, memorized = verifier.verify_candidates(candidates)

    # Speichern
    with open(args.output, "w") as f:
        json.dump({
            "total_candidates": len(candidates),
            "memorized_count": len(memorized),
            "memorized": memorized,
            "all_verified": verified
        }, f, indent=2, ensure_ascii=False)

    print(f"\nMemorized: {len(memorized)}/{len(candidates)}")
    for m in memorized:
        print(f"Score: {m['similarity_score']} | URL: {m['best_match_url']}")
        print(f"Text: {m['text'][:100]}\n")

if __name__ == "__main__":
    main()