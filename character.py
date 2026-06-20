
from __future__ import annotations
import weakref
import random
from typing import ClassVar
from dataclasses import dataclass, field
from logging import getLogger

_logger: ClassVar[logging.Logger] = getLogger(__name__)

@dataclass(eq=False)
class Character:

    name: str
    is_npc: bool            = False
    is_narrator: bool       = False
    is_silent: bool         = False
    wants_to_speak: float   = 0.0
    pool: int               = 0
    relationships: dict[str, float] = field(default_factory=dict)
    beliefs: dict[str, str] = field(default_factory=dict)
    io_handle: Any          = None  # FastAPI websocket etc
    character_card: str     = ""  # injected into prompt


def renormalize(subroster:[Character])-> None:
    total: float = 0.0
    for c in subroster:
        total += c.wants_to_speak

    if abs(total) > .001:
        for c in subroster:
            c.wants_to_speak -= total / len(subroster)

        
def choose_speaker(subroster:[Character], suppress_update:bool = False) -> Character:

    winner: Character  = None
    winning_score: float

    _logger.debug("choose_speaker()")

    for c in subroster:
        _logger.debug("consider character: " + repr(c))
        if (winner is None) or (c.wants_to_speak > winning_score):  # TODO(ben) add randomness
            winner = c
            winning_score = c.wants_to_speak

    if (winner is not None) and (not suppress_update):
        winner.wants_to_speak -= 5.0 # TODO(ben) magic number should be parameter of some sort
        renormalize(subroster)
        
    return winner
        



# 1. Character class with wants-to-speak and speaker selection
# 2. Narrator as special character (validates the framework)
# 3. Single presence pool (everyone together, simplest case)
# 4. Multi-pool support
# 5. Dynamic pool membership
