# import sys
# from llama_cpp import Llama
# import llama_cpp
# import inspect
import copy
from config import Config
from llm_manager import LlmManager
import logging

# print("\n----------Llama------------")
# print(inspect.signature(Llama))


def main():
    config = Config.load()

    setup_logging(config.log)

    logger = logging.getLogger("config")
    print("----config parameters:------\n" + repr(config))
    logger.debug("config logger is printing.")
    
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

def setup_logging(log_directives: list[str]) -> None:
    """Configures system log levels based on uniform strings.
    
    Accepts: ["parser=debug", "llm=info,main=warning", "critical"]
    """
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    # Basic baseline configuration for the global root logger
    logging.basicConfig(level=logging.WARNING, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    # Normalize entries to support both comma-separated lists and standard strings
    flat_directives = []
    for entry in log_directives:
        flat_directives.extend([item.strip() for item in entry.split(",") if item.strip()])

    for directive in flat_directives:
        if "=" in directive:
            # Component target override (e.g., "parser=debug")
            logger_name, level_str = directive.split("=", 1)
            logger_name = logger_name.strip()
            level_str = level_str.strip().lower()

            if level_str not in level_map:
                print(f"Error: Unknown log level '{level_str}' for component '{logger_name}'.", file=sys.stderr)
                print(f"Valid options are: {', '.join(level_map.keys()).upper()}", file=sys.stderr)
                sys.exit(1)

            logging.getLogger(logger_name).setLevel(level_map[level_str])
        else:
            # Global catch-all directive override (e.g., "debug")
            level_str = directive.strip().lower()
            if level_str in level_map:
                logging.getLogger().setLevel(level_map[level_str])
            else:
                print(f"Error: Unknown global log level '{level_str}'.", file=sys.stderr)
                sys.exit(1)

# end of setup_loggimg()

if __name__ == "__main__":
    main()
