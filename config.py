from dataclasses import dataclass

@dataclass
class Config:
    MODEL_ROOT = "/mnt/models_nvme/models"
    MODEL_PATH = (f"{MODEL_ROOT}/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-IQ4_XS.gguf")

    system_behavior_prompt_file = "tb/system_prompt_test.txt"
    system_formatting_prompt_file = "/dev/null"
    initial_prompt_file = "./prompts/init.txt"
    temperature = 0  # for debug
    context_size = 4096
    ngl = 999

    max_sample_len=200

    with open(system_behavior_prompt_file, "r") as f:
        system_behavior_prompt = " ".join(f.read().splitlines())

    with open(system_formatting_prompt_file, "r") as f:
        system_formatting_prompt = " ".join(f.read().splitlines())

    with open(initial_prompt_file, "r") as f:
        initial_prompt = " ".join(f.read().splitlines())

    verbose: bool      = False  # to avoid spewage of library internal info into output stream
    model_path: str    = MODEL_PATH
    n_ctx: int         = context_size
    n_gpu_layers: int  = ngl
    flash_attn: bool   =True
    seed: int          = 1

    top_k: int               = 1 
    top_p: float             = 1.0 
    min_p: float             = 0.0 
    typical_p: float         = 1.0 
    temp: float              = 0.0 
    repeat_penalty: float    = 1.0 
    frequency_penalty: float = 0.0 
    presence_penalty: float  = 0.0 
    tfs_z: float             = 1.0 
    mirostat_mode: int       = 0 
    mirostat_eta: float      = 0.1 
    mirostat_tau: float      = 5.0 
    penalize_nl: bool        = True 
    logits_processor= None #Optional[LogitsProcessorList]
    grammar= None #Optional[LlamaGrammar]
    idx= None #Optional[int]
    
