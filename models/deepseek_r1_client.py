
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class DeepSeekClient:
    """ Class defining the DeepSeek client. Orignally intended to be the original DeepSeek R1 model,
    but ressources available were not sufficient, so deepseek-ai/DeepSeek-R1-Distill-Qwen-14B was chosen instead.
    If needed and possible, the model of course can be chosen directly by the model_name parameter (deepseek-ai/DeepSeek-R1)."""
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
           # device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16 
        ).to(self.device)
        
        self.model.eval()
        
    def generate(self, prompt="", max_new_tokens=256, temperature=1.0, top_k=40):
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