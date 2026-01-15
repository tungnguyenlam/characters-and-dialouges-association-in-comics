from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List
import torch
import gc

class BaseModel(ABC):
    """Base class for all models in the pipeline."""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        verbose: bool = False,
        plot: bool = False,
        model_name: str = ""
    ):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.verbose = verbose
        self.plot = plot
        self.model = None
        self.model_name = model_name

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != 'auto':
            return device
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    @property
    def is_loaded(self) -> bool:
        return self.model is not None
    
    def _check_loaded(self):
        if not self.is_loaded:
            raise RuntimeError(f"{self.model_name}: Model not loaded. Call load_model() first.")
    
    def _log(self, message: str):
        if self.verbose:
            print(f"[{self.model_name}] {message}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load model into memory."""
        pass
    
    def unload_model(self) -> None:
        """Free model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            
            if 'cuda' in self.device:
                torch.cuda.empty_cache()
            elif 'mps' in self.device:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            self._log("Model unloaded")
    
    def preprocess(self, inputs: Any) -> Any:
        """Override for custom preprocessing."""
        return inputs
    
    def postprocess(self, outputs: Any) -> Any:
        """Override for custom postprocessing."""
        return outputs
    
    @abstractmethod
    def _inference(self, inputs: Any, **kwargs) -> Any:
        """Core inference logic. Must be implemented by subclass."""
        pass
    
    def predict(
        self, 
        inputs: Any, 
        skip_preprocess: bool = False,
        skip_postprocess: bool = False,
        **kwargs
    ) -> Any:
        """Main prediction interface."""
        self._check_loaded()
        
        if not skip_preprocess:
            inputs = self.preprocess(inputs)
        
        outputs = self._inference(inputs, **kwargs)
        
        if not skip_postprocess:
            outputs = self.postprocess(outputs)
        
        if self.plot:
            self._visualize(inputs, outputs)
        
        return outputs
    
    def _visualize(self, inputs: Any, outputs: Any) -> None:
        """Override for custom visualization."""
        pass
    
    def __enter__(self):
        """Context manager: auto load."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: auto unload."""
        self.unload_model()
    
    def __repr__(self) -> str:
        return f"{self.model_name}(device={self.device}, loaded={self.is_loaded})"