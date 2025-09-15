from abc import ABC, abstractmethod

class ModelWrapper(ABC):
    """
    Abstract base class for model wrappers.
    """
    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def training_step(self):
        pass

    @abstractmethod
    def validation_step(self):
        pass

    @abstractmethod
    def test_step(self):
        pass
    
    @abstractmethod
    def visualize_step(self):
       pass
    
   