
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
        )
        self._jinja_template_str = self._llm.metadata.get("tokenizer.chat_template")
        self.logger.info("chat template: " + self._jinja_template_str)
        self._jinja_template = Template(self._jinja_template_str)

        # decode the token IDs directly from your loaded model
        self.bos_str = self._llm._model.token_get_text(self._llm.token_bos())
        self.eos_str = self._llm._model.token_get_text(self._llm.token_eos())

        
    def generate_chat_reply(self, messages: list[dict[str, str]], config: Config, stream: bool = True) -> str:
        '''Generate a response for a chat-formatted message list'''

        current_prompt_token_list: list[int]
        sampled_token : int
        sampled_tokens: list[int]

        self.logger.debug("-------raw message block--------\n" + repr(messages) + "--------------------")
        self.logger.info(f"final entry in messages block: {messages[-1]}")

        rendered = self._jinja_template.render(
            messages=messages,
            add_generation_prompt=True,
            bos_token=self.bos_str,
            eos_token=self.eos_str,
            tools=None,
            strftime_now=lambda fmt: datetime.now().strftime(fmt)
        )

        self.logger.debug("-------rendered message block--------\n" + repr(rendered) + "--------------------")

        current_prompt_token_list = self._llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)

        # facilitates correlation tests against llama-cli.  this is a
        # diagnostic function only.  This output can be used by -f in a
        # llama-cli run to bypass chat_template processing.  Also
        # can run tokenize_my_prompt_with_llama to see raw token list as
        # tokenized by llama-cli.
        # intentionally strip off the leading BOS because llama-cli adds
        # that even if you tell it to use no-cnv mode.
        if config.dump_prompts:
            with open("rendered_prompt.txt", "w") as f:
                f.write(rendered[len(self.bos_str):])
            with open("tokenized_prompt.txt", "w") as f:
                f.write(repr(current_prompt_token_list))


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
            if stream == True:
                print(self._llm.detokenize([sampled_token]).decode("utf-8", errors="ignore"), end="", flush=True)

        reply = self._llm.detokenize(sampled_tokens).decode("utf-8", errors="ignore")
        self.logger.info(f"reply: -{reply}-")
        return reply

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
        self.logger.debug(f"_sync_kv_cache: cached={len(self._cached_tokens)}, new={len(new_tokens)}, match={match_len}")
        if match_len < len(self._cached_tokens):
            self._rollback(match_len)
        self._eval(new_tokens[match_len:])

# end class LlmManager
