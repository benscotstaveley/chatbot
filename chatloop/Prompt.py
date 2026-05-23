from enum import Enum

class PromptFieldUpdateType(Enum):
    APPEND
    OVERWRITE


class Prompt:

    _kv_cache_end : int
    
    def __init__(self):
        _kv_cache_end = 0
