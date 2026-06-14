
from llama_cpp import Llama
from config import Config
from jinja2 import Template
from logging import getLogger
from datetime import datetime


class LlmManager:
    '''Wraps the Llama object, to keep low-level KV cache in sync with python representation of it'''

    logger = getLogger(__name__)
    
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
            verbose=False, #TODO(ben): once logging is set up
            model_path=config.model,
            n_ctx=config.ctx,
            n_gpu_layers=config.ngl,
            flash_attn=config.flash_attn,
            seed=config.seed,

            top_k=config.top_k,
            top_p=config.top_p,
            min_p=config.min_p,
            typical_p=config.typical_p,
            temp=config.temp,
            repeat_penalty=config.repeat_penalty,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            tfs_z=config.tfs_z,
            mirostat_mode=config.mirostat_mode,
            mirostat_eta=config.mirostat_eta,
            mirostat_tau=config.mirostat_tau,
            penalize_nl=config.penalize_nl,
            logits_processor=None,  # Optional[LogitsProcessorList]
            grammar=None,  # Optional[LlamaGrammar]
            idx=None,  # Optional[int]
        )
        self._jinja_template_str = self._llm.metadata.get("tokenizer.chat_template")
        self.logger.info("chat template: " + self._jinja_template_str)
        self._jinja_template = Template(self._jinja_template_str)

    def generate_chat_reply(self, messages: list[dict[str, str]], config: Config = None, display: bool = True) -> str:
        '''Generate a response for a chat-formatted message list'''

        current_prompt_token_list: list[int]
        sampled_token : int
        sampled_tokens: list[int]

        self.logger.debug("-------raw message block--------\n" + repr(messages) + "--------------------")

        # decode the token IDs directly from your loaded model
        model_bos = self._llm.detokenize([self._llm.token_bos()]).decode("utf-8")
        model_eos = self._llm.detokenize([self._llm.token_eos()]).decode("utf-8")

        bos_str = self._llm._model.token_get_text(self._llm.token_bos())
        eos_str = self._llm._model.token_get_text(self._llm.token_eos())

        rendered = self._jinja_template.render(
            messages=messages,
            add_generation_prompt=True,
            #bos_token=model_bos,
            #eos_token=model_eos,
            bos_token=bos_str,
            eos_token=eos_str,
            tools=None,
            strftime_now=lambda fmt: datetime.now().strftime(fmt)
        )

        # enable this file creation for correlation tests (against llama-cli).
        # then run tokenize_my_prompt_with_llama
        # with open("/tmp/test_prompt.txt", "w") as f:
        #     f.write(rendered[len("<|begin_of_text|>"):])

        self.logger.debug("-------rendered message block--------\n" + repr(rendered) + "--------------------")

        current_prompt_token_list = self._llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)

        self.logger.debug(f"tokenized prompt({len(current_prompt_token_list)}): " + repr(current_prompt_token_list))
        self.logger.debug("parameters at the time of calling the sampler: " + repr(config))
        # perform prefill, using as much of the former prompt as possible.  this
        # updates the _cached_tokens list.
        self._sync_kv_cache(current_prompt_token_list)

        # _sampler accumulates repeat penalty state across calls,
        # so it must be reset for each independent generation
        self._llm._sampler = self._llm._init_sampler(
            top_k=config.top_k,
            top_p=config.top_p,
            min_p=config.min_p,
            typical_p=config.typical_p,
            temp=config.temp,
            repeat_penalty=config.repeat_penalty,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            tfs_z=config.tfs_z,
            mirostat_mode=config.mirostat_mode,
            mirostat_tau=config.mirostat_tau,
            mirostat_eta=config.mirostat_eta,
            penalize_nl=True,
            logits_processor=None,
            grammar=None,
        )

        # now do the sample/eval loop
        sampled_tokens = []
        while True:
            sampled_token = self._sample()
            self.logger.debug(f"sampled[{len(sampled_tokens)}]: id={sampled_token} text='{self._llm.detokenize([sampled_token]).decode('utf-8', errors='ignore')}'")
            if (sampled_token == self._llm.token_eos()) or (len(sampled_tokens) >= config.max_sample_len):
                break
            sampled_tokens.append(sampled_token)
            self._eval([sampled_token])

        return self._llm.detokenize(sampled_tokens).decode("utf-8", errors="ignore")

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
        print(f"_sync_kv_cache: cached={len(self._cached_tokens)}, new={len(new_tokens)}, match={match_len}")
        if match_len < len(self._cached_tokens):
            self._rollback(match_len)
        self._eval(new_tokens[match_len:])

# end class LlmManager
