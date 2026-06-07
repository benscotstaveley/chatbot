import copy
from config import Config
from llm_manager import LlmManager
from loop import loop
import logging
import os

# define BOOTSTRAP_DEBUG to debug pre-logger codepaths

def main():
    # Basic baseline configuration for the global root logger
    # TODO(ben): Replace with QueueHandler / QueueListener deferred logging 
    # to cleanly buffer early logs before config files are loaded.
    # Quick bootstrap check for parsing phase
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG if os.getenv("BOOTSTRAP_DEBUG") == "1" else logging.WARNING,
                        format=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    config = Config.load()

    setup_logging(config.log)

    logger = logging.getLogger(__name__)
    logger.info("chatbot starting.")
    logger.debug("config parameters:" + repr(config))
    
    llm_manager = LlmManager(config)

    loop(
        config=config,
        llm_manager=llm_manager
    )

# end of main

def setup_logging(log_directives: list[str]) -> None:
    """Configures system log levels based on uniform strings.
    
    Accepts: ["parser=debug", "llm=info,main=warning", "critical"]
    """

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
