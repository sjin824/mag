from abc import ABC, abstractmethod
import gc
import logging

class BaseHandler(ABC):
    def __init__(self, config: dict = None):
        self.config = config
        self.service = None
        self.device = None 
        self._set_logger()
        
    def set_device(self, device):
        self.device = device
        
    def _set_logger(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def release_resources(self):
        del self.service
        self.service = None
        gc.collect()
        self.logger.info(f"{self.__class__.__name__} resources released.")
    
    # =================== Need to be implemented in the child class.  =================== 
    @abstractmethod
    def load_service(self):
        pass
    
    @abstractmethod
    def _formatter(self, batch):
        pass

    @abstractmethod
    def _process_logic(self, formatted_batch):
        pass
    
    # ====================================================================================
    
    # Encapsulates the process_logic method with error handling and logging.
    def process(self, batch):
        self.logger.info(f"Processing inputs with {self.__class__.__name__}")
        try:
            formatted_batch = self._formatter(batch)
            results = self._process_logic(formatted_batch)
            self.logger.info(f"Processing complete.")
            return results
        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}: {str(e)}", exc_info=True)
            raise