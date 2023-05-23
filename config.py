import os

class Config:
    def __init__(self) -> None:
        self.device = os.getenv("DEVICE", "cuda")
        self.model_vicuna = os.getenv("MODEL_VICUNA", "helloollel/vicuna-7b")
        self.num_gpus = os.getenv("NUM_GPUS", 1)
        self.load_8bit = True if os.getenv("LOAD_8BIT") else False
        self.max_new_tokens = os.getenv("MAX_NEW_TOKENS", 512)
        self.temperature = os.getenv("TEMPERATURE", 1.0)


    def __str__(self) -> str:
        return str(self.__dict__)