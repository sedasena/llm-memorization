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
        self.tokenizer.padding_side = "left" # bc we are generating, need to pad left

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
         #   device_map="auto",
            dtype=torch.float16 #if self.device=="cuda" else torch.float32
        ).to(self.device)
        
        self.model.eval()
        
    def generate(self, prompt="", max_new_tokens=256, temperature=1.0, top_k=40):
        # einzelner String
        if not prompt:
            input_ids = torch.tensor(
                [[self.tokenizer.eos_token_id]], device=self.device
            )
        else:
            input_ids = self.tokenizer(
                prompt, return_tensors="pt"
            )["input_ids"].to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    
    
    def calculate_perplexity(self, text):
        input_ids = torch.tensor(
            self.tokenizer.encode(text)
        ).unsqueeze(0).to(self.device)
    
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
    
        loss, logits = outputs[:2]
        return torch.exp(loss).item()