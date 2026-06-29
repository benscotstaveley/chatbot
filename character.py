
from __future__ import annotations
import weakref
import random
from typing import ClassVar
from dataclasses import dataclass, field
from logging import getLogger
from enum import Enum
import re
import sys
import json
from pathlib import Path

_logger: ClassVar[logging.Logger] = getLogger(__name__)

SpeakerRole = Enum("SpeakerRole", ["NARRATOR", "SILENT", "HUMAN", "NPC"] )
    
@dataclass(eq=False)
class Character:

    name: str
    role: SpeakerRole
    wants_to_speak: float   = 0.0
    talkativeness: float    = 0.5   # 0-1 scale
    pool: int               = 0
    relationships: dict[str, float] = field(default_factory=dict)
    beliefs: dict[str, str] = field(default_factory=dict)
    io_handle: Any          = None  # FastAPI websocket etc
    character_card: str     = ""  # injected into prompt

def format_wts(subroster: {Character})-> str:
    formatted:str = ""
    for c in subroster:
        if formatted != "":
            formatted += "; "
            
        formatted += f"{c.name}: {c.wants_to_speak}"

    return formatted
        
def renormalize(subroster:{Character})-> None:
    minimal: float = None
    maximal: float = None
    for c in subroster:
        if (minimal is None) or (c.wants_to_speak < minimal):
            minimal = c.wants_to_speak
        if (maximal is None) or (c.wants_to_speak > maximal):
            maximal = c.wants_to_speak

    if (maximal - minimal) > .001:
        for c in subroster:
            c.wants_to_speak = (c.wants_to_speak - minimal) / (maximal - minimal)

def initialize_speaking_order(subroster: {Character})-> None:
    for c in subroster:
        if c.role == SpeakerRole.NARRATOR:
            c.wants_to_speak = 1.0
        else:
            c.wants_to_speak = 0.0

def update_speaking_order_for_delay(subroster: {Character})-> None:
    for c in subroster:
        c.wants_to_speak += c.talkativeness

    renormalize(subroster)
    

def choose_speaker(subroster:{Character}, suppress_update:bool = False) -> Character:

    winner: Character  = None
    winning_score: float

    _logger.debug("choose_speaker()")

    for c in subroster:
        _logger.debug("consider character: " + repr(c))
        if (winner is None) or (c.wants_to_speak > winning_score):  # TODO(ben) add randomness
            winner = c
            winning_score = c.wants_to_speak

    if (winner is not None) and (not suppress_update):
        winner.wants_to_speak -= len(subroster)
        renormalize(subroster)
        
    return winner
        
def increase_mentioned_character_speaking_priority(subroster:{Character}, speech: str) -> None:
    for c in subroster:
        if bool(re.search(rf'\b{re.escape(c.name)}\b', speech, re.IGNORECASE)):
            c.wants_to_speak += 0.4
            _logger.debug(f"raising WTS for {c.name} because character was mentioned by current speaker")
    renormalize(subroster)
 
        
def initialize_roster(roster: set(Character), roster_file:str)-> None:
    roster.add(Character(name="NARRATOR", role=SpeakerRole.NARRATOR, talkativeness=0.0))
    roster.add(Character(name="(pause)",  role=SpeakerRole.SILENT, talkativeness=0.1))

    path = Path(roster_file)
    if not path.is_file():
        _logger.error(f"character roster file not found: {roster_file}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            character_list = json.load(f)
            for character in character_list:
                 _logger.debug(f"adding character {character}")
                 # sanitize
                 character_name : str | None = character.get("name")
                 if character_name is None:
                     _logger.error(f"reading roster: character {character} does not have a 'name'")
                     continue
                 character_role : str | None = character.get("role")
                 if character_role is None:
                     _logger.error(f"reading roster: character {character} does not have a 'role'")
                     continue
                 character_role_enum: SpeakerRole
                 try:
                     character_role_enum = SpeakerRole[character_role.upper()]
                 except KeyError:
                     _logger.error(f"reading roster: character {character} has a role '{character_role}' that is not understood.")
                     continue
                 
                 # TODO(ben) more sanity checking; extract more fields
                 roster.add(Character(name=character_name, role=character_role_enum))

            _logger.info(f"final roster: {roster}")
    except json.JSONDecodeError as e:
        _logger.error(f"error reading character roster file {roster_file}: {e}")
        return{}
    except OSError as e:
        _logger.error(f"OS error trying to open character roster file {roster_file}: {e}")
        return {}

# 1. Character class with wants-to-speak and speaker selection
# 2. Narrator as special character (validates the framework)
# 3. Single presence pool (everyone together, simplest case)
# 4. Multi-pool support
# 5. Dynamic pool membership
