from enum import Enum

class SpeakerAgentType(Enum):
    LLM
    HUMAN
    SILENT
    NARRATOR

class GlobalState:

    # List of all messages spoken from the beginning of time
    # actually need dict b/c we need to know who the speaker was.
    # that also gives us a trail to get the mood of the speaker when s/he said it
    message_history: [str]

    # number of characters tracked
    num_characters: int # >= 0
    
    # the first entry in the List that is in the literal-speech context window
    window_head: int
    
    def __init__(self):
        self.characters = {}
        self.narration = []
        self.transcript = []
    

    def update(proposal: XXX) -> UpdateResult:
        raise NotImplementedError("This method is not yet implemented")


class Relationship:
    # some thoughts:
    # friendliness vs loyalty: f. can be insincere or formal, or out of fear or social pressure
    # fear, respect: i'm thinking of dropping one.  the purpose is a 2-stage evaluation: use
    #   respect when the character has a choice, but fear is always a hard-stop baseline.  a char
    #   will harm a respected person only if forced, but never a feared char
    # all Relationships XYZ are R[i][j]: i's XYZ j
    # so, for instance, 'respect for' in entry [i][j] means "i's respect for j"
    friendliness_towards: int #0-9, 5 is neutral
    loyalty_towards: int  #0-9  # 0-9;  0-9;  is neutral.  <5 indicates active desire to harm
    fear_of: int
    respect_for: int
    social_dominance_over: int # could be boss/worker, parent/child, rockstar/dumbass, or whatever
    
class CharacterState:
    # this is a long tail.  what do we leave freeform rather than categorize?  do we include
    # 'tiredness'?  'confusion'? 'angry'?
    name: str
    freeform_description: str
    is_present: bool  # is the character currently present and participating in dialog?
    talkativeness: int
    boldness: int
    
# TODO: how do we use this?  what do we know upon creation?  do we update?  when?
# we maintain a list of CharacterState (initial and final) for all characters, and
# likewise an importance rank for each character, not just the speaker.  Consider
# a turn in which character A accuses character B of treason.  Although A is the
# speaker, the turn is important to B as well, and both characters' Relationship
# scores will change.  A turn is not just about the speaker.
class TurnState:
    speaker_agent_type: SpeakerAgentType
    speaker: int     # within the category of SpeakerAgentType
    initial_state: [CharacterState]
    final_state: [CharacterState]
    dialog: str
    importance: [int]    # importance of this turn for each character.
                         # for each, 0-9. usually because of some large change in state
    
