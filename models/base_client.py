from abc import ABC, abstractmethod

## Base client as an interface for the models
class BaseClient(ABC):
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def generate(self, max_new_tokens=256):
        pass
    
    @abstractmethod
    def perplexity(self, text: str):
        pass