import sys
from llama_cpp import Llama
import llama_cpp
import inspect
import copy
from jinja2 import Template
from config import Config
from llm_manager import LlmManager

print("\n----------Llama------------")
print(inspect.signature(Llama))

config = Config() ;
print("-----init object-----\n" + repr(config))
llm_obj = LlmManager(config)

print("----llama_cpp version: " + llama_cpp.__version__)
print(dir(llm_obj._llm))

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

messages = [
    {"role": "system", "content": config.system_behavior_prompt + config.system_formatting_prompt},
]

llm_object = LlmManager(config)

while True:
    
    user_input = input("USER: ")
    messages.append({"role": "user", "content": user_input})

    print("-----messages-----" + repr(messages) + "------------------")

    # construct a prompt specific for brief query
    messages_iter = copy.deepcopy(messages)
    messages_iter[-1]["content"] += " Be brief."
    reply_brief = llm_object.generate_chat_reply(messages_iter, config)
    print("-------brief reply:-------\n" + reply_brief)


    # construct a prompt specific for detailed query
    messages_iter = copy.deepcopy(messages)
    messages_iter[-1]["content"] += " Describe in detail."
    reply_detailed = llm_object.generate_chat_reply(messages_iter, config)
    print("-------detailed reply:-------\n" + reply_detailed)

    messages.append({"role": "assistant", "content": reply_brief})

# end of while forever



#print("user input, re-detokenized:-"+llm.detokenize(tokens).decode("utf-8", errors="ignore")+"-")

