import copy
from logging import getLogger
from config import Config
from llm_manager import LlmManager

def loop(config:Config, llm_manager:LlmManager):

    logger = getLogger(__name__)
    logger.info("starting chat loop")

    messages = [
        {"role": "system", "content": config.system_behavior_prompt + config.system_formatting_prompt},
    ]

    logger.info("messages list after adding system prompt:" + repr(messages))

    turn_number:int = 0
    while True:

        # Human
        if turn_number==0:
            user_input = config.initial_prompt
        else:
            user_input = input("USER: ")
        messages.append({"role": "user", "content": user_input})

        logger.debug("-----messages-----" + repr(messages) + "------------------")

        # # construct a prompt specific for brief query
        # messages_iter = copy.deepcopy(messages)
        # messages_iter[-1]["content"] += " Be brief."
        # reply_brief = llm_manager.generate_chat_reply(messages_iter, config)
        # logger.debug("-------brief reply:-------\n" + reply_brief)

        # # construct a prompt specific for detailed query
        # messages_iter = copy.deepcopy(messages)
        # messages_iter[-1]["content"] += " Describe in detail."
        # reply_detailed = llm_manager.generate_chat_reply(messages_iter, config)
        # logger.debug("-------detailed reply:-------\n" + reply_detailed)

        reply = llm_manager.generate_chat_reply(messages, config)
        print(reply)
        messages.append({"role": "assistant", "content": reply})

        turn_number += 1
    # end of while forever
