from torch import device
import torch
from .llama3_1_8b_client import LlamaClient
from transformers import AutoTokenizer, AutoModelForCausalLM
#from huggingface_hub import login
#login()
#import os
#os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

class GemmaClient(LlamaClient):
    def __init__(self, model_name="google/gemma-3-4b-it", device=None):
        super().__init__(model_name=model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
    
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
         #   device_map="auto",
            dtype=torch.float16 #if self.device=="cuda" else torch.float32
        ).to(self.device)
        
        self.model.eval()
        
    
    def generate(self, prompt="", max_new_tokens=256, temperature=1.0, top_k=40):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k
        )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def perplexity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()