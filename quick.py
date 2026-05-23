import sys
from llama_cpp import Llama
import inspect

max_sample_len = 256

MODEL_ROOT = "/home/ben/llama.cpp/models"

# MODEL_PATH = (f"{MODEL_ROOT}/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_M.gguf")
MODEL_PATH = (f"{MODEL_ROOT}/l3.2-rogue-7b/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-IQ4_XS.gguf")

system_behavior_prompt_file = "/dev/null"
system_formatting_prompt_file = "/dev/null"
initial_prompt_file = "./prompts/init.txt"
temperature = 0  # for debug
context_size = 4096
ngl = 24

with open(system_behavior_prompt_file, "r") as f:
    system_behavior_prompt = " ".join(f.read().splitlines())

with open(system_formatting_prompt_file, "r") as f:
    system_formatting_prompt = " ".join(f.read().splitlines())

with open(initial_prompt_file, "r") as f:
    initial_prompt = " ".join(f.read().splitlines())

print("Initializing the Llama")
llm = Llama(
    verbose=False,  # to avoid spewage of library internal info into output stream
    model_path=MODEL_PATH,
    n_ctx=context_size,
    n_gpu_layers=ngl,
    flash_attn=True,
)

# for first-time devel
# print("\n\n----------eval------------\n")
# print(inspect.signature(llm.eval))
# print("\n\n----------sample------------\n")
# print(inspect.signature(llm.sample))
# print("\n\n----------tokenize------------\n")
# print(inspect.signature(llm.tokenize))
# print("\n\n----------detokenize------------\n")
# print(inspect.signature(llm.detokenize))
# sys.exit()

prompt = system_behavior_prompt + system_formatting_prompt + initial_prompt

print("tokenize...\n")
tokens = llm.tokenize(prompt.encode("utf-8"))

# prompt evaluation (prefill)
print("prefill...\n")
llm.eval(tokens)

# generation
print("generate...\n")
sampled_tokens = []
while True:

    sampled_token = llm.sample(
        top_k=40, #int
        top_p=0.95, #float
        min_p=0.05, # float
        typical_p=1.0, # float
        temp=0.8, #float
        repeat_penalty=1.0, #float
        frequency_penalty=0.0, #float
        presence_penalty=0.0, #float
        tfs_z=1.0, #float
        mirostat_mode=0, #int
        mirostat_eta=0.1, #float
        mirostat_tau=5.0, #float
        penalize_nl=True, #bool
        logits_processor=None, #Optional[LogitsProcessorList]
        grammar=None, #Optional[LlamaGrammar]
        idx=None, #Optional[int]
    )
    if (sampled_token == llm.token_eos()) or (len(sampled_tokens) > max_sample_len):
        break
    sampled_tokens.append(sampled_token)
    llm.eval([sampled_token])
    print(f"sapled token: {sampled_token}")

generated_text = llm.detokenize(sampled_tokens).decode("utf-8", errors="ignore")

print(generated_text)
