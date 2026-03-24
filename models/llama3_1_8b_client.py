import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_client import BaseClient

class LlamaClient(BaseClient):
    def __init__(self,
        model_name="meta-llama/Llama-3.1-8B",
        device=None):
        super().__init__(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
        #    device_map="auto",
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


    """def generate(self, prompt="", max_new_tokens=256, temperature=1.0, top_k=40):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k
        )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)"""
    
    def calculate_perplexity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()
    
    
        