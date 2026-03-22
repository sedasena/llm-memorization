from .llama3_1_8b_client import LlamaClient
from .deepseek_r1_client import DeepSeekClient
from .qwen3_1_7b_client import QwenClient
from .gemma_client import GemmaClient



class MultiModelClient:
    def __init__(self):
        self.models = {
            "llama": LlamaClient(),
            "deepseek": DeepSeekClient(),
            "qwen": QwenClient(),
            "gemma": GemmaClient()
        }

    def generate(self, model_name="llama"):
        return self.models[model_name].generate()
    
    def perplexitites(self, text):
        return {
            name: model.perplexity(text)
            for name, model in self.models.items()
        }