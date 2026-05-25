import sys
from llama_cpp import Llama
import inspect

max_sample_len = 200

MODEL_ROOT = "/mnt/models_nvme/models"
MODEL_PATH = (f"{MODEL_ROOT}/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-IQ4_XS.gguf")

system_behavior_prompt_file = "/dev/null"
system_formatting_prompt_file = "/dev/null"
initial_prompt_file = "./prompts/init.txt"
temperature = 0  # for debug
context_size = 4096
ngl = 999

with open(system_behavior_prompt_file, "r") as f:
    system_behavior_prompt = " ".join(f.read().splitlines())

with open(system_formatting_prompt_file, "r") as f:
    system_formatting_prompt = " ".join(f.read().splitlines())

with open(initial_prompt_file, "r") as f:
    initial_prompt = " ".join(f.read().splitlines())

print(f"Initializing the Llama.  seed={sys.argv[1]}")
llm = Llama(
    verbose=False,  # to avoid spewage of library internal info into output stream
    model_path=MODEL_PATH,
    n_ctx=context_size,
    n_gpu_layers=ngl,
    flash_attn=True,
    seed=int(sys.argv[1]),

        top_k=1, #int
        top_p=1.0, #float
        min_p=0.0, # float
        typical_p=1.0, # float
        temp=0.0, #float
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

llm._sampler = llm._init_sampler(
    top_k=1,
    top_p=1.0,
    min_p=0.0,
    typical_p=1.0,
    temp=0.0,
    repeat_penalty=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    tfs_z=1.0,
    mirostat_mode=0,
    mirostat_tau=5.0,
    mirostat_eta=0.1,
    penalize_nl=True,
    logits_processor=None,
    grammar=None,
)

#for first-time devel
# print("\n----------Llama------------")
# print(inspect.signature(Llama))
# print("\n----------eval------------")
# print(inspect.signature(llm.eval))
# print("\n----------dir(llm)------------")
# print(dir(llm))
# print("\n----------sample------------")
# print(inspect.signature(llm.sample))
# print("\n----------tokenize------------")
# print(inspect.signature(llm.tokenize))
# print("\n----------detokenize------------")
# print(inspect.signature(llm.detokenize))
# sys.exit()

prompt = system_behavior_prompt + system_formatting_prompt + initial_prompt

print("tokenize...")
tokens = llm.tokenize(prompt.encode("utf-8"))

print("---------prompt tokens to be input to eval()----------")
print(repr(tokens))


# prompt evaluation (prefill)
print("prefill...")
llm.eval(tokens)

# generation
print("generate...")
sampled_tokens = []
while True:

    sampled_token = llm.sample()
    sampled_text = llm.detokenize([sampled_token]).decode("utf-8", errors="ignore")
    # print(f"sampled token: {sampled_token} (text: -{sampled_text}-)")
    if (sampled_token == llm.token_eos()) or (len(sampled_tokens) >= max_sample_len):
        break
    sampled_tokens.append(sampled_token)
    # print(f"n_tokens={llm.n_tokens};  llm.n_tokens: {llm.n_tokens}")
    
    # manually maintain input_ids so repeat penalty works
    llm.input_ids[llm.n_tokens-1] = sampled_token
    
    llm.eval([sampled_token])

print("---------sampled tokens--------")
print(repr(sampled_tokens))
generated_text = llm.detokenize(sampled_tokens).decode("utf-8", errors="ignore")
print(generated_text)


