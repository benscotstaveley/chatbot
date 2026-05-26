import sys
from llama_cpp import Llama
import llama_cpp
import inspect
from jinja2 import Template

class LlmManager:
    def generate(self, prompt: str, display: bool) -> str:
        #internally calls _render → _sync_kv_cache → _eval → _sample_loop

    def _eval(self, tokens: list[int]) -> None:
        self._llm.eval(tokens)
        self._kv_tokens.extend(tokens)

    def _rollback(self, position: int) -> None:
        self._llm._ctx.kv_cache_seq_rm(0, position, -1)
        self._llm.n_tokens = position
        self._kv_tokens = self._kv_tokens[:position]

    def _sample(self) -> int:
        return self._llm.sample()
    
    def _sync_kv_cache(self, new_tokens: list[int]) -> None:
        match_len = 0
        for a, b in zip(self._kv_tokens, new_tokens):
            if a != b:
                break
            match_len += 1
        if match_len < len(self._kv_tokens):
            self._rollback(match_len)
        self._eval(new_tokens[match_len:])

# end class LlmManager

def get_turn_tokens(llm, t, role, content):
    # render a fake two-turn conversation where the second turn is what we want
    rendered = t.render(
        messages=[
            {"role": "user", "content": "X"},
            {"role": role, "content": content},
        ],
        add_generation_prompt=(role != "assistant"),
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>",
        tools=None,
    )
    # tokenize the full render, then subtract the tokens for just "X" turn
    baseline = t.render(
        messages=[{"role": "user", "content": "X"}],
        add_generation_prompt=False,
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>",
        tools=None,
    )
    full_tokens = llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)
    base_tokens = llm.tokenize(baseline.encode("utf-8"), add_bos=False, special=True)
    # the new turn tokens are the suffix after the baseline
    return full_tokens[len(base_tokens):]

# end of def get_turn_tokens


max_sample_len = 200

MODEL_ROOT = "/mnt/models_nvme/models"
MODEL_PATH = (f"{MODEL_ROOT}/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-IQ4_XS.gguf")

system_behavior_prompt_file = "tb/system_prompt_test.txt"
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

# Note for later: Also note for later — kv_cache_seq_rm and kv_cache_seq_cp are both there, which is exactly what we'll need for Step 2's bookmark/rollback mechanism. Good to know they're accessible.

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

# Jinja-encode the system and initial prompts
print("jinja...")
template_str = llm.metadata.get("tokenizer.chat_template")
t = Template(template_str)

messages = [
        {"role": "system", "content": system_behavior_prompt + system_formatting_prompt},
        {"role": "user",   "content": initial_prompt}
    ]
# messages = [
#         {"role": "system", "content": system_behavior_prompt + system_formatting_prompt},
#         {"role": "user",   "content": initial_prompt},
#         {"role": "assistant",   "content": "Located in the northern region of Île-de-France, Paris is a city with a rich history and culture that dates back to the Roman era. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.  Paris has been the capital of France since 987, when it was made the seat of the French monarchy. Over the centuries, the city has played a significant role in European and world history, being a major center of politics, art, and culture.  Today, Paris is a bustling metropolis with over 2.1 million people living within its city limits, and more than 12 million people living in the metropolitan area. The city is known for its fashion, cuisine, and tourism, attracting millions of visitors each year.  Some of the popular attractions in Paris include: * The Eiffel Tower: an iconic iron lattice tower that was built for the 1889 World"},
#         {"role": "user",   "content": "and what is its population?"}
#     ]

print(f"-------initial message------\n"+repr(messages)+"\n----------")
rendered = t.render(
    messages=messages,
    add_generation_prompt=True,
    bos_token="<|begin_of_text|>",
    eos_token="<|eot_id|>",
    tools=None,
)
print("-------rendered system plus initial prompt--------\n"+repr(rendered)+"\n-----------")

tokens = llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)

for iter in [0,1]:

    # at the top of the loop, 'tokens' is the incremental query, either from the
    # code above, in which case it is turn 0 (system prompt + initial prompt),
    # or from the prior execution of the loop, in which case it is the user
    # input from the prior turn.

    # prefill
    print(f"prefill ({len(tokens)} tokens, at n_tokens={llm.n_tokens}): "+ repr(tokens))
    llm.eval(tokens)
    bookmark = llm.n_tokens
    print(f"bookmark set at n_tokens={bookmark}")

    # --- query 1 ---
    suffix_1 = "<|start_header_id|>user<|end_header_id|>\n\nQuery 1: brief answer please.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    suffix_1_tokens = llm.tokenize(suffix_1.encode("utf-8"), add_bos=False, special=True)
    llm.eval(suffix_1_tokens)

    print(f"after eval suffix_1, n_tokens={llm.n_tokens}")

    llm._sampler = llm._init_sampler(
        top_k=1, top_p=1.0, min_p=0.0, typical_p=1.0,
        temp=0.0, repeat_penalty=1.1,
        frequency_penalty=0.0, presence_penalty=0.0,
        tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0,
        mirostat_eta=0.1, penalize_nl=True,
        logits_processor=None, grammar=None,
    )
    sampled_tokens = []
    print("----------\ndetokenize of input_ids:"+repr(llm.detokenize(list(llm.input_ids[:llm.n_tokens]))))
    while True:
        sampled_token = llm.sample()
        print(f"  sampled: {sampled_token} = -{llm.detokenize([sampled_token]).decode('utf-8', errors='ignore')}-")
        if (sampled_token == llm.token_eos()) or (len(sampled_tokens) >= max_sample_len):
            break
        sampled_tokens.append(sampled_token)
        llm.eval([sampled_token])

    print(f"QUERY 1 sampled_tokens: {repr(sampled_tokens)}")
    print(f"after eval response_1, n_tokens={llm.n_tokens}")

    result_1 = llm.detokenize(sampled_tokens).decode("utf-8", errors="ignore")
    print(f"QUERY 1 RESULT: {result_1}")

    # --- rollback to bookmark ---
    llm._ctx.kv_cache_seq_rm(0, bookmark, -1)
    old_n_tokens = llm.n_tokens
    llm.n_tokens = bookmark
    print(f"rolled back from n_tokens={old_n_tokens} to n_tokens={bookmark}, n_tokens now {llm.n_tokens}")

    # --- query 2 ---
    suffix_2 = "<|start_header_id|>user<|end_header_id|>\n\nQuery 2: elaborate in detail please.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    suffix_2_tokens = llm.tokenize(suffix_2.encode("utf-8"), add_bos=False, special=True)
    llm.eval(suffix_2_tokens)

    print(f"after eval query_2, n_tokens={llm.n_tokens}")

    llm._sampler = llm._init_sampler(
        top_k=1, top_p=1.0, min_p=0.0, typical_p=1.0,
        temp=0.0, repeat_penalty=1.1,
        frequency_penalty=0.0, presence_penalty=0.0,
        tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0,
        mirostat_eta=0.1, penalize_nl=True,
        logits_processor=None, grammar=None,
    )
    sampled_tokens = []
    while True:
        sampled_token = llm.sample()
        if (sampled_token == llm.token_eos()) or (len(sampled_tokens) >= max_sample_len):
            break
        sampled_tokens.append(sampled_token)
        llm.eval([sampled_token])

    print(f"after eval suffix_2, n_tokens={llm.n_tokens}")

    result_2 = llm.detokenize(sampled_tokens).decode("utf-8", errors="ignore")
    print(f"QUERY 2 RESULT: {result_2}")

    user_input = input("USER: ")

    # at this point, query_1 has been discarded;  result_1 has been sent to stdout
    # and otherwise discarded; query_2 and result_2 are in the KV cache and
    # n_tokens accounts for these.
    # now we must render the user input; that will then be tokenized at the
    # top of the loop for the next iteration.
    tokens = get_turn_tokens(llm, t, "user", user_input)

    print("user input, re-detokenized:-"+llm.detokenize(tokens).decode("utf-8", errors="ignore")+"-")

sys.exit()

while True:   # turn loop

    print("-------messages------\n"+repr(messages)+"\n----------")
    rendered = t.render(
        messages=messages,
        add_generation_prompt=True,
        bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>",
        tools=None,
    )
    #print("-------rendered prompt--------\n"+repr(rendered)+"\n-----------")

    tokens = llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)

    # reset KV cache and position
    llm._ctx.kv_cache_clear()
    llm.n_tokens = 0

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

    # prompt evaluation (prefill)
    print(f"prefill...\n" + repr(tokens) + "\n")
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
    
        llm.eval([sampled_token])

        #print("---------sampled tokens--------"+repr(sampled_tokens))
        generated_text = llm.detokenize(sampled_tokens).decode("utf-8", errors="ignore")
        #print(generated_text)

    # end of sample loop
    
    # append assistant response to message history
    messages.append({"role": "assistant", "content": generated_text})

    print(f"ASSISTANT: {generated_text}")

    # get user input, tokenize, and append
    user_input = input("USER: ")

    messages.append({"role": "user", "content": user_input})

# end of turn loop


