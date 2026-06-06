# import sys
# from llama_cpp import Llama
# import llama_cpp
# import inspect
import copy
from config import Config
from llm_manager import LlmManager

# print("\n----------Llama------------")
# print(inspect.signature(Llama))


def main():
    config = Config.load()

    print("----config parameters:------\n" + repr(config))

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
# end of main


if __name__ == "__main__":
    main()
