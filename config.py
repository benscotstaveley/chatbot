from __future__ import annotations
import argparse
from dataclasses import dataclass, field, fields
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, get_type_hints
from logging import getLogger

logger = getLogger(__name__)


@dataclass
class Config:

    # --- File Paths & Defaults ---
    model: str = None
    system_behavior_prompt_file: str   = "prompts/system.txt"
    system_formatting_prompt_file: str = "/dev/null"
    initial_prompt_file: str           = "prompts/init.txt"
    roster_file: str                   = "prompts/roster.txt"
    
    # --- Diagnostic Modes ---
    dump_prompts: bool = False
    single_shot: bool = False

    # --- Runtime Parameters ---
    ctx: int = 4096
    ngl: int = 999
    flash_attn: bool = True
    seed: int = 1
    max_sample_len: int = 500

    # --- Sampling Parameters ---
    top_k: int = 1
    top_p: float = 1.0
    min_p: float = 0.0
    typical_p: float = 1.0
    temp: float = 0.0
    repeat_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    tfs_z: float = 1.0
    mirostat_mode: int = 0
    mirostat_eta: float = 0.1
    mirostat_tau: float = 5.0
    penalize_nl: bool = True

    # --- Derived File Contents (populated at instantiation, omitted from init) ---
    log: List[str] = field(default_factory=list)

    system_behavior_prompt: str = field(default="", init=False)
    system_formatting_prompt: str = field(default="", init=False)
    initial_prompt: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Populates the string prompts once the exact file paths are settled."""
        self.system_behavior_prompt = self._read_prompt(
            self.system_behavior_prompt_file
        )
        self.system_formatting_prompt = self._read_prompt(
            self.system_formatting_prompt_file
        )
        self.initial_prompt = self._read_prompt(self.initial_prompt_file)

        if not self.model:
            raise ValueError("model path must be specified via config file or --model flag")

    @staticmethod
    def _read_prompt(path_str: str) -> str:
        """Safely reads a prompt file if it exists, stripping linebreaks."""
        path = Path(path_str)
        if str(path) == "/dev/null" or not path.is_file():
            logger.debug("empty file: " + str(path))
            return ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return " ".join(f.read().splitlines())
        except OSError:
            logger.error("error reading " + str(path))
            return ""

    @classmethod
    def load(cls) -> Config:
        """Factory Method implementing the primary configuration chain:

        1. Compile-time defaults (built into dataclass)
        2. Home directory file override (~/.config/toyconfig.json)
        3. Current working directory file override (./.config/toyconfig.json)
        4. Command line flag override
        """
        config_dict: Dict[str, Any] = {}

        # 2. Layer: Home Directory
        home_config = Path.home() / ".config" / ".chatconfig.json"
        logger.debug(f"process home_config:{home_config}")
        update_dict =  cls._deserialize(home_config)
        logger.debug(f"file {home_config}: merging " + repr(update_dict) + " into " + repr(config_dict) )
        cls._merge_dict(config_dict, update_dict)

        # 3. Layer: Current Working Directory
        cwd_config = Path.cwd() / ".config" / ".chatconfig.json"
        logger.debug(f"process cwd_config:{cwd_config}")
        update_dict =  cls._deserialize(cwd_config)
        logger.debug(f"file {cwd_config}: merging " + repr(update_dict) + " into " + repr(config_dict) )
        cls._merge_dict(config_dict, update_dict)

        # 4. Layer: Command Line
        logger.debug("process command line")
        update_dict = cls._parseargs()
        logger.debug("cmdline: merging " + repr(update_dict) + " into " + repr(config_dict) )
        cls._merge_dict(config_dict, update_dict)
        logger.debug("final non-default config parameters: " + repr(config_dict))
        
        # Returns perfectly resolved instance, prompting __post_init__ safely *once*
        return cls(**config_dict)

    @classmethod
    def _merge_dict(cls, base: dict, overrides: dict) -> None:
        """Mutates the base dictionary with valid dataclass inputs."""
        valid_fields = {f.name for f in fields(cls) if f.init}
        for key, value in overrides.items():
            if key not in valid_fields:
                logger.warning(f"unknown config key ignored: '{key}'")
                continue
            if isinstance(base.get(key), list) and isinstance(value, list):
                base[key] = base[key] + value
            else:
                base[key] = value

    @staticmethod
    def _deserialize(path: Path) -> dict:
        """Reads a JSON configuration layer file."""
        if not path.is_file():
            logger.debug("file not found:" + str(path))
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.debug(f"results of json.load(): {data}")
                # Safeguard: if json defines "log" as a single string, normalize it to a list
                if isinstance(data, dict) and "log" in data and isinstance(data["log"], str):
                    data["log"] = [data["log"]]
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError as e:
            logger.error(f"json.JSONDecodeError in {path}: {e}")
            return {}
        except (OSError):
            logger.error("OS error trying to open file for deserialize")
            return {}

    @classmethod
    def _parseargs(cls) -> dict:
        """Dynamically builds CLI parser from initialized dataclass fields."""
        parser = argparse.ArgumentParser(description="LLM Runner Configuration")

        # This automatically converts "int" back into int, "bool" back into bool, etc.
        type_hints = get_type_hints(cls)

        for f in fields(cls):
            if not f.init:
                continue

            # Replace underscores with hyphens for clean command line switches
            cli_name = f.name.replace("_", "-")
            field_type = type_hints[f.name] # Safe, true Python type object
            # Check if the field is a List type (or typing.List)
            # Origin checks for generic aliases like list or List
            is_list = getattr(field_type, "__origin__", None) is list or field_type is list

            if field_type is bool:
                # Add dual switches to explicitly declare truthiness
                parser.add_argument(f"--{cli_name}", action="store_true")
                parser.add_argument(f"--no-{cli_name}", action="store_true")
            elif is_list:
                # This allows passing --log multiple times
                parser.add_argument(f"--{cli_name}", action="append")
            else:
                parser.add_argument(f"--{cli_name}", type=field_type)

        # Suppress defaults so argparse doesn't inject fields the user missed
        # directly over lower-precedence JSON config layers
        for action in parser._actions:
            if action.dest != "help":
                action.default = argparse.SUPPRESS

        namespace = parser.parse_args(sys.argv[1:])
        provided = vars(namespace)
        resolved_bools: Dict[str, Any] = {}

        # Reconstruct fields and resolve custom boolean logic
        for f in fields(cls):
            if not f.init:
                continue

            # Check what name argparse assigned to the destination property
            dest_name = f.name
            field_type = type_hints[f.name] # Safe, true Python type object

            if field_type is bool:
                # Use hyphens for the negative switch lookups
                no_dest = f"no_{f.name}"

                if dest_name in provided:
                    resolved_bools[dest_name] = True
                elif no_dest in provided:
                    resolved_bools[dest_name] = False
            elif dest_name in provided:
                resolved_bools[dest_name] = provided[dest_name]

        return resolved_bools
