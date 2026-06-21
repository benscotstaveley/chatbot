
from __future__ import annotations
import weakref
import random
from typing import ClassVar
from dataclasses import dataclass, field
from logging import getLogger
from enum import Enum

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
        



# 1. Character class with wants-to-speak and speaker selection
# 2. Narrator as special character (validates the framework)
# 3. Single presence pool (everyone together, simplest case)
# 4. Multi-pool support
# 5. Dynamic pool membership
