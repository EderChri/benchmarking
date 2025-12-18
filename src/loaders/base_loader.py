from abc import ABC, abstractmethod

LOADER_REGISTRY = {}

def register_loader(source_name: str):
    def decorator(cls):
        LOADER_REGISTRY[source_name] = cls
        return cls
    return decorator

class BaseDataLoader(ABC):
    def __init__(self, config, test_split_ratio=0.2):
        self.config = config
        self.test_split_ratio = test_split_ratio
    
    @abstractmethod
    def load(self):
        pass
