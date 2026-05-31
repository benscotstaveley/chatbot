
from llama_cpp import Llama
from config import Config
from jinja2 import Template

class LlmManager:
    '''Wraps the Llama object, to keep low-level KV cache in sync with python representation of it'''

    # the Llama LLM object
    _llm: Llama
    
    # the _cached_tokens[] list is always in sync with what is in the HW KV cache
    _cached_tokens: list[int]

    # the actual jinja script from the model, in jinja2 language
    _jinja_template_str: str

    # the jinja template used to render chat messages; created once at init time
    _jinja_template: Template | None
    
    def __init__(self, config: Config) -> None:
        self._cached_tokens = []
        self._llm = Llama(
            verbose = config.verbose,
            model_path = config.model_path,
            n_ctx = config.n_ctx,
            n_gpu_layers = config.n_gpu_layers,
            flash_attn = config.flash_attn,
            seed = config.seed,

            top_k = config.top_k,
            top_p = config.top_p,
            min_p = config.min_p,
            typical_p = config.typical_p,
            temp = config.temp,
            repeat_penalty = config.repeat_penalty,
            frequency_penalty = config.frequency_penalty,
            presence_penalty = config.presence_penalty,
            tfs_z = config.tfs_z,
            mirostat_mode = config.mirostat_mode,
            mirostat_eta = config.mirostat_eta,
            mirostat_tau = config.mirostat_tau,
            penalize_nl = config.penalize_nl,
            logits_processor=None, #Optional[LogitsProcessorList]
            grammar=None, #Optional[LlamaGrammar]
            idx=None, #Optional[int]
        )
        #print("\n\n-----llama dir:\n")
        #print( dir(_llm))
        #print("jinja...")
        self._jinja_template_str = self._llm.metadata.get("tokenizer.chat_template")
        self._jinja_template = Template(self._jinja_template_str)
        # print(self._template_str)
        
    def generate_chat_reply(self, messages: list[dict[str, str]], config: Config = None, display: bool = True) -> str:
        '''Generate a response for a chat-formatted message list'''

        current_prompt_token_list: [int]
        sampled_token : int
        sampled_tokens: [int]

        print("-------message block--------\n" + repr(messages) + "--------------------")

        rendered = self._jinja_template.render(
            messages=messages,
            add_generation_prompt=True,
            bos_token="<|begin_of_text|>",
            eos_token="<|eot_id|>",
            tools=None,
        )
        current_prompt_token_list = self._llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)

        # perform prefill, using as much of the former prompt as possible.  this
        # updates the _cached_tokens list.
        self._sync_kv_cache(current_prompt_token_list)

        # TODO(ben) i'm not 100% clear on why we need to do this.
        self._llm._sampler = self._llm._init_sampler(
            top_k= config.top_k,
            top_p= config.top_p,
            min_p= config.min_p,
            typical_p= config.typical_p,
            temp= config.temp,
            repeat_penalty= config.repeat_penalty,
            frequency_penalty= config.frequency_penalty,
            presence_penalty= config.presence_penalty,
            tfs_z= config.tfs_z,
            mirostat_mode= config.mirostat_mode,
            mirostat_tau= config.mirostat_tau,
            mirostat_eta= config.mirostat_eta,
            penalize_nl=True,
            logits_processor=None,
            grammar=None,
        )

        # now do the sample/eval loop
        sampled_tokens = []
        while True:
            sampled_token = self._sample()
            if (sampled_token == self._llm.token_eos()) or (len(sampled_tokens) >= config.max_sample_len):
                break
            sampled_tokens.append(sampled_token)
            self._eval([sampled_token])

        return(self._llm.detokenize(sampled_tokens).decode("utf-8", errors="ignore"))

    def _eval(self, tokens: list[int]) -> None:
        self._llm.eval(tokens)
        self._cached_tokens.extend(tokens)

    def _rollback(self, position: int) -> None:
        self._llm._ctx.kv_cache_seq_rm(0, position, -1)
        self._llm.n_tokens = position
        self._cached_tokens = self._cached_tokens[:position]

    def _sample(self) -> int:
        return self._llm.sample()
    
    def _sync_kv_cache(self, new_tokens: list[int]) -> None:
        match_len = 0
        for a, b in zip(self._cached_tokens, new_tokens):
            if a != b:
                break
            match_len += 1
        if match_len < len(self._cached_tokens):
            self._rollback(match_len)
        self._eval(new_tokens[match_len:])

# end class LlmManager
