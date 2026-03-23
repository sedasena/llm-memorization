from ddgs import DDGS
from sentence_transformers import SentenceTransformer, util
import time

class SBertVerifier:
    """ This class defines the SBERT verifier for automatic verification of the generated results by the models,
    by searching automatically on duckduckgo search. """
    def __init__(self, sbert_model="all-MiniLM-L6-v2", threshold=0.85):
        self.model = SentenceTransformer(sbert_model)
        self.threshold = threshold
        
    def search(self, query, num_results=5):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query[:200],
                    max_results=num_results
                ))
                return results
        except Exception as e:
            print(f"There as been a search error: {e}")
            return []
        
    def verify(self, text):
        query = text[:100].strip()
        results = self.search(query)
        
        if not results:
            return {
                "is_memorized": False,
                "similarity_score": 0.0,
                "best_match_url": None,
                "best_match_text": None
            }
        # Convert the generated text to an embedding
        generated_embedding = self.model.encode(
            text, convert_to_tensor=True
        )
        
        best_score = 0.0
        best_match = None
        for result in results:
            text_snippet = result.get("body", "")
            if not text_snippet:
                continue
            
            snippet_embedding = self.model.encode(text_snippet, convert_to_tensor=True)
            # calculate the cosine-similarity between the generated text emb. and the
            # snippet embedding of the result (body) text embedding of the search query
            score = util.cos_sim(generated_embedding, snippet_embedding).item()
            
            if score > best_score:
                best_score = score
                best_match = result
        # the generated text is deemed memorized if the final best score is passing the threshold defined
        is_memorized = (best_score >= self.threshold)
        
        return {
            "is_memorized": is_memorized,
            "similarity_score": round(best_score, 4),
            "best_match_url": best_match["href"] if best_match else None,
            "best_match_text": best_match["body"] if best_match else None
        }
        
    def verify_candidates(self, candidates, text="text", delay=2.0):
        verified = []
        
        for i, cand in enumerate(candidates):
            result = self.verify(cand[text])
            verified.append({**cand, **result})
            # to avoid duckduckgo rate limit
            time.sleep(delay) 
        
        memorized = [r for r in verified if r["is_memorized"]]
        print(f"\n Result: {len(memorized)}/{len(candidates)} memorized")
        return verified, memorized


verifier = SBertVerifier()

# Testfall: ein bekannter Text der online auffindbar sein sollte
test_text = "The MIT License is a permissive free software license originating at the Massachusetts Institute of Technology"

result = verifier.verify(test_text)
print(result)