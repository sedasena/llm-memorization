
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class DeepSeekClient:
    
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float16 if self.device=="cuda" else torch.float32
        )
        
        self.model.eval()
        
    def generate(self, prompt="", max_new_tokens=256, temperature=1.0, top_k=40):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
        output_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_k=top_k
        )
            
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
    def perplexity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        outputs = self.model(
            **inputs,
            labels=inputs["input_ids"]
        )
        
        loss = outputs.loss
        return torch.exp(loss).item()