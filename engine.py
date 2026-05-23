import sys
import os
import re
from enum import Enum, auto
from typing import NewType, TypeDict, NamedTuple, DefaultDict, Optional
from llama_cpp import Llama, ChatCompletionRequestMessage
from chatloop import parser
from chatloop.State import State
from collections import defaultdict

class CharacterType(Enum):
    HUMAN = auto()
    NARRATOR = auto()
    SILENT = auto()
    NPC = auto()


@dataclass
class Relationship:
    love: int
    respect: int
    fear: int
    trust: int

@dataclass(frozen=True)
class Character:
    idx: int            # unique identifying index.  bookkeeping only.
    character_type : CharacterType
    name: str
    description: str
    presence_group: int
    talkativeness: int
    boldness: int
    friendliness: int
    volatility: float   # used for next_speaker calculation

class Roster:
    """
    """
    roster: list[Character]
    
    def __init(self)__ -> None:
        self.roster = None

    def add(new_character_type: CharacterType, new_character: Character) -> int:

# TODO(ben) need beliefs matrix


        
@dataclass
class TurnState:
    speaker_type : CharacterType
    speaker : int
    intent : str    # short; used for gatekeeping
    dialog : str    # long flowery speech to be seen by user
    importance : int   # used by compactor; usualluy denotes large change in state
    goal_short_term : str
    goal_long_term : str
    # below here, things start to be specific to the scenario
    

UPDATE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\.(\w+)(?:\[(\w+)\])?\s*=\s*(.+)$")

MODEL_ROOT = "/mnt/models_nvme/models"

# MODEL_PATH=f"{MODEL_ROOT}/Mistral-Small-24B-ArliAI-RPMax-v1.4.Q5_K_M.gguf"
# MODEL_PATH=f"{MODEL_ROOT}/dolphin-2.6-mistral-7b.Q6_K.gguf"
# MODEL_PATH=f"{MODEL_ROOT}/Qwen2.5-32b-RP-Ink-Q4_K_M.gguf"
# MODEL_PATH=f"{MODEL_ROOT}/Mistral-rp-24b-karcher.i1-Q5_K_M.gguf"
# MODEL_PATH=f"{MODEL_ROOT}/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-IQ4_XS.gguf"
MODEL_PATH = (
    f"{MODEL_ROOT}/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_M.gguf"
)
system_behavior_prompt_file = "./sys_behavior.txt"
system_formatting_prompt_file = "./sys_formatting.txt"
initial_prompt_file = "../init.txt"
temperature = 0  # for debug
context_size = 24578
ngl = 24

# this works in conjunction with Llama(verbose=False) to suppress messages
# into stdout/stderr from within the library itself
os.environ["LLAMA_LOG_LEVEL"] = "ERROR"

# Context manager to silence C-level stderr
class SuppressStderr:
    def __enter__(self):
        self.stderr_fd = sys.stderr.fileno()
        self.saved_stderr_fd = os.dup(self.stderr_fd)
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, self.stderr_fd)

    def __exit__(self, *args):
        os.dup2(self.saved_stderr_fd, self.stderr_fd)
        os.close(self.devnull)
        os.close(self.saved_stderr_fd)



class SpeakerArbiter:

    # maintain a List that maps a character index to priority (float)
    _weights : DefaultDict[int, DefaultDict[int, Optional[float]]] = \
        defaultdict(lambda : defaultdict(lambda : None))
    
    def __init__(self) -> None:
        self.my_data: DefaultDict[int, DefaultDict[int, Optional[float]]] = \
            defaultdict(lambda: defaultdict(lambda: None))           

        # add all characters

        _renormalize()
        
    def choose(randomness:float = 1.0) -> int :
        """
        Determine who will speak next.
        """
        # add a bit of randomness to each character, according to that character's
        # Volatility

        # pick the largest as the next speaker

        # renormalize the list

    def apply_recenty_penalty(character: int) -> None:
        """
        Lower the priority of the given speaker to account for the fact that
        he or she has recently spoken.  The amount to lower is equal to the
        number of characters at this point in time.  This number doesn not
        include the silent speakers or narrators
        """

    def adjust(character: int, adjustment: float)->None:

    def add_new_character(new_character: int, initial_priority: float)-> None:
        # TODO(ben) how to handle adding/removing a character later in the chat?
        # else use pseudorandom selection

    def _renormalize()->None:

# note to self: i don't know if i want the roster to be simply a dict or a class.
# the argument for class is that if someday we want to support adding a character
# during gameplay, we need to remember to do other things like extend the
# SpeakerArbiter, the Relationship matrix, etc.  Maybe we'll never allow this
# functionality?

def main():
    # 1. SETUP: read prompts; Initialize the model

    # TODO(ben): we may want to make some of this machine-generated, like
    # character cards.  we may want to pass a list of percentages
    # into the LLM to create a distilled personality

    with open(system_behavior_prompt_file, "r") as f:
        system_behavior_prompt = " ".join(f.read().splitlines())

    with open(system_formatting_prompt_file, "r") as f:
        system_formatting_prompt = " ".join(f.read().splitlines())

    with open(initial_prompt_file, "r") as f:
        initial_prompt = " ".join(f.read().splitlines())

    state = State()

    # note to self: i chose regular dict for this to prioritize crash over
    # silent wrong execution if we access a missing entry
    RelationshipMatrix: dict[dict[Relationship]]

    def add_relationship_to_matrix(new_character: int,
                                   new_character_towards_others: dict[int, Relationship],
                                   others_towards_new_character: dict[int, Relationship]) -> None:
        # add new row

        # add column by adding to all pre-existing entries
    
    print("Initializing Llama")
    llm = Llama(
        verbose=False,  # to avoid spewage of library internal info into output stream
        model_path=MODEL_PATH,
        n_ctx=context_size,
        n_gpu_layers=ngl,
        flash_attn=True,
    )
    #  with SuppressStderr():
    #      llm = Llama(
    #          model_path=MODEL_PATH,
    #          n_ctx=24576,      # Your preferred context size
    #          n_gpu_layers=20,  # -1 means offload everything to GPU
    #          flash_attn=True,   # High speed optimization for newer commits
    #          #n_batch=1,
    #          #no_kv_offload=False,
    #      )

    print("--- Roleplay Engine loop starting ---")

    messages = [
        {
            "role": "system",
            "content": system_behavior_prompt + system_formatting_prompt,
        },
        {"role": "user", "content": initial_prompt},
    ]

    # new loop:

    # TODO(ben) set up inital characters
    
    game_turn = 0
    state_history : [State] = []
    speakers = SpeakerArbiter(character_list)

    # at the start of the game, we set initial conditions that will lead to
    # a clean start of the turn sequence: first human speaker goes first as
    # a mechanism of establishing the initial prompt, and the narrator goes
    # next to set the scene for the human character(s), who may not know the
    # initial prompt.

    # TODO(ben) check that at least one human and at least one narrator
    # exist, and do something appropriate if not.  this is hacky; what is
    # the right way to do this?
    
    speakers.adjust((Human, 0), 100)
    speakers.adjust((Narrator, 0), 50)

    current_speaker : int

    
    while True:

        current_speaker = speakers.choose()

        # for now, we always take the arbiter's choice
        speakers.apply_recency_penalty(current_speaker)

        # at the top of the loop we have n-1's unprocessed human or NPC output.
        # record what happens for HHMMH to cover all transitions.

        if current_speaker.character_type == HUMAN :

            # we may or may not have an intent (if n-1 was LLM we do; else (human or none) we do not)
            # we may or may not have the flowery prose; if human or LLM we do; if NONE we do not
            # we need to compute all state updates
            # we need to check the state updates
            # we need to apply the state updates
            
            human_output : str
            # this will become parallel thread A, though today we serialize
            # get human's output
            if game_turn == 0:
                human_output = intitial_prompt
                human_dialog = process_slash_commands(human_output)

                # at this point we have human dialog from interactive turn

            else :
                # TODO(ben) someday support input from network socket, etc.
                while True :
                    human_output = input(f"[{current_speaker.name}]: ")
                    if len(human_output) == 0 :
                        break  # totally blank line is legitimate dialog
                    human_dialog = process_shash_commands(human_output)
                    if len(human_dialog) > 0 :
                        break  # this is a line of dialog b/c text left after processing slash commands

                # at this point we have human dialog from interactive turn
            # at this point we have human dialog from interactive or initial turn
            
            canonicalized_human_output = canonicalize_human_input(human_output)

            # this will become parallel thread B, though today we serialize
            # while human is outputting (our input), process prior turn into state

            # for first turn there is no n-1 intent/prose to update into state
            # if the narrator spoke last there is nothing further to render
            if game_turn == 0:

                # special-case first turn.  just as we got the first turn's speech from a
                # text file input, so we get the first turn's state updates (the initial
                # state treated as deltas applied to a zero state)
                
            else if game_turn>0 and state_history[game_turn].character_type != NARRATOR:
                while True :
                    delta_query : str = render_state_to_delta_query() # get prompt to use to ask for state deltas
                    get_detailed_deltas()  # actually run the prompt.
                    parse_deltas()  # into CharacterState, Relationship, Belief
                    if successful_parse or give_up:
                        break
                    
                record_authoratative_history()

            if context_compression_required() :
                compress_context()
                
            # someday when we parallelize A and B: wait for both threads to finish

        else : # else not human player turn

            # first turn goes to human, to load initial prompt and state.  we
            # check here because we want to assume there was an n-1
            assert turn != 0

            # we have the flowery prose
            # there may or may not be intent (present iff prior was LLM)
            # we do not have n-1's state deltas
            # we have not recorded n-1's latest state.
            # we must compute n-1's state deltas from the flowery prose
            # we must then compute n's starting state
            # then we generate intent
            
            # get a brief description of what the llm wants to do.  low-temp query.
            get_npc_intent()


        # note we could be computing quick deltas from human or NPC output
        # render_state_to_quick_deltas
        # get_quick_deltas().  (later overwrite with the detailed update that happens in parallel with human output
        #   this is deltas that affect the chosen NPC only)
        # record_quick_history()
        # get_npc_intent(): 
        # swap_kv_cache()  # save latest flowery KV cache to system RAM; install KV cache appropriate for intent
        # for intent_iteration 0..max_before_giving_up
        # render_state_to_query_intent(speaker, global_state, intent_iteration)
        # llama.eval():  # accumulate the entire intent output
        # canonicalize_llm_output # this function correlated with format convenient for llms to output
        # scrub(canonicalized_reply, policies).  both quick python checks and quick yes/no call to LLM
        # either break intent_iteration loop or repeat loop to try again.  handle "too many attempts" somehow.

        # call the LLM with normal temp to convert the intent to flowery
        # prose.  this takes some time and we stream directly to the user.
        # it is tempting to do one more scrub after this point but 1. we have
        # already scrubbed the intent, and a deviation from intent at this point
        # is unlikely, and is properly fixed by changing the prompt we give at
        # this point.  also, 2. this is a long-latency operation and we don't
        # want to put it all into TTFT.

        # end of loop.  iterate.

        # (**) Notes on the undo feature:  this would ordinarily be used to undo the
        # last output of the LLM.  But we can unwind all the way back to the start if
        # requested.  This is especially useful because the LLM may be selected several
        # times before the human gets a chance to enter this command.
        # Note 2: my use of "intent" aligns with what others refer to as "thought block".
        # I think they basically serve the same purpose.

        # some topics not to forget: post-mortem reveal (how much is under user control).
        # ability to go back to any point in time (undo actually does this).

        break  # so the loop won't keep doing nothing while it's just comments
        # note to self: break call to LLM into intent and then prose for many reasons:
        # 1. intent is quick and we can start the discriminator on that instead of the
        # final prose with good confidence.  maybe we can even just stream the prose in
        # realtime; (2) run with different temps (3) makes parsing a bit easier
        # note: if intent stored in "intent", the call to the prose LLM can be
        # [Stage direction: {{intent}}] [Speaker name]:

        # other notes: keeping the intent (a/k/a thoughtblock) is also useful for
        # a postmortem, and for debug, and also to feed to the discriminator, because
        # not everything will come through in the prose.  especially if the intent
        # is to lie.  Also, look for too many consecutive intents.  even if the
        # prose is different each time this will feel dull.

        # more notes: if we have "presence pools" instead of just IsPresent, meaning "with
        # the human", we open up possibilities of NPCs cooperating out of hearing of
        # the player, which would be pretty cool.

        # note: on pass n through the loop, authoritative_turn contains the last turn
        # number whose detailed prose was used in calculating the ongoing Relationships,
        # Beliefs, and CharacterState data (as opposed to all turns except the prior
        # one that have had their 'intent' block referenced).  Every NPC turn we update
        # with n-1's state, and every human turn we do an update going back to
        # authoritative_turn+1 up to and including n-1, compute more reliable state,
        # compare with what we had just for checking if approximate state is wrong, and
        # overwriting the approximate state.
        # initial state: first pass we force selection of human, and

        # OH, wait.... the human input must be taken in full before the NPCs play...
        # NPC after human: full update based on full human input
        # NPC after NPC: quick update based on intent of n-1
        # human after NPC: here, instead of just updating with intent of n-1, we
        #   do a full update going back to the last time human-after-npc did this
        # human after human: this doesn't really happen.  we should do a full update.

        # note the human parts of the state really should be maintained by the human.

        # more notes: user commands:
        #           case /quit: quit the loop; end game
        #           case /temperature: change temp
        #           case /undo: un-commit the last update to global state.  see notes below (**)
        #           case /save
        #           case /load
        #           case /reveal (to end game or cheat)
        #           case maybe someday: in-game edits of game state (god mode, debugging/profiling/optimizing)
        #           default break do loop; exit with user input to go into history as verbatim speech
    while True:
        # 1. Generate response

        print(
            "---------about to call create_chat_completion() with the following:---------"
        )
        print(messages)
        print(
            "----------------------------------------------------------------------------"
        )

        # we would like to have this hook for debug, but it is unavailable in
        # our version of the library
        # print("----and here is the formatted prompt that will be passed to the model:------")
        # print(llm._format_chat_prompt(messages))
        # print("----------------------------------------------------------------------------")

        full_reply = ""
        for chunk in llm.create_chat_completion(
            stream=True, messages=messages, temperature=temperature, max_tokens=500
        ):

            delta = chunk["choices"][0]["delta"]
            finish_reason = chunk["choices"][0].get("finish_reason", None)

            if "content" in delta:
                text = delta["content"]
                print(text, end="", flush=True)
                full_reply += text

            if finish_reason in ["length"]:
                # TODO(ben): we want to check for runaway text, and if that's not the
                # case, continue fetching text.  But this doesn't seem to work:
                # the next chunk comes back NULL and terminates the for loop
                print("\ngot 'length' termination; continue fetching response text\n")
            if finish_reason in ["stop"]:
                # TODO(ben): handle these cases better.  for now, break the loop
                break

        print()

        state = parser.parse_llm_output(full_reply, state)

        # Parse structured updates
        # if the model sent a state update block, and we can unambiguously understand
        # it, and it passes various extra sanity tests, commit it to RP state.  this
        # step should be conservative.
        update_block = extract_state_update(full_reply)
        if update_block:
            state = parse_state_update(update_block, state)

        # 3. Get user input
        user_input = input("\nYou: ")

        # process user commands:
        # /quit: quit the loop
        # /temperature: change temp
        # /discard: discard model output and re-query
        # /save
        # /load
        # TODO(ben): also, discard last user input, if that's head of queue

        # 4. Append to history
        state.narration.append("USER: " + user_input)
        state.transcript.append(full_reply)

        messages.append({"role": "assistant", "content": full_reply})
        messages.append({"role": "user", "content": user_input})


# end of main()
from typing import List, Sequence
#BEGIN SAMPLE CODE
class ConversationContext:
    def __init__(self, system_prompt_tokens: List[int]):
        # The internal mutable ground truth for your active context window
        self._tokens: List[int] = list(system_prompt_tokens)
        
        # The exact token index up to which the KV cache is currently valid
        self._kv_cache_dirty_index: int = len(self._tokens)

    @property
    def tokens(self) -> Sequence[int]:
        """Expose tokens safely as a read-only Sequence to external code."""
        return self._tokens

    @property
    def kv_cache_index(self) -> int:
        return self._kv_cache_dirty_index

    def append_tokens(self, new_tokens: List[int]) -> None:
        """Appends tokens. KV cache pointer remains intact; it just needs to advance on next eval."""
        self._tokens.extend(new_tokens)
        # Note: Do NOT automatically advance kv_cache_index here. 
        # It stays where it was, signaling to your LLM loop exactly where it needs to resume evaluating.

    def apply_digest(self, digest_start: int, digest_end: int, summary_tokens: List[int]) -> None:
        """
        Replaces a block of historical turns with a summary digest.
        Invalidates the KV cache pointer immediately from the point of modification.
        """
        # In-place slice replacement
        self._tokens[digest_start:digest_end] = summary_tokens
        
        # CRITICAL: Any modification to history breaks the KV cache from that point forward.
        # The KV cache must be rewound or recalculated starting at the digest point.
        self._kv_cache_dirty_index = min(self._kv_cache_dirty_index, digest_start)

    def advance_kv_cache(self, processed_count: int) -> None:
        """Call this after llama.eval() succeeds to sync the cache pointer."""
        self._kv_cache_dirty_index += processed_count

    def get_str(self, tokenizer) -> str:
        """Exposes the entire current context as string for high-level APIs."""
        return tokenizer.decode(self._tokens)
# END SAMPLE CODE
class Prompt :
    final_turn_in_digest : int
    final_turn_in_current_state : int
    current_full_propmt : str     # includes KV-cache preamble plus new stuff at end
    noncached_ptr : int    # index into current_full_prompt of first byte NOT in KV cache

    def __init__(self):
        final_turn_in_digest = -1
        final_turn_in_current_state = -1
        

# this will become a method.
# TODO: list all prompt transformatinos;
# H: - ask for a delta for all (intents? and) spoken since last H call.  
def render_query_for_npc_intent(state: State) -> str:
    raise NotIMplementedError("Not Implemented")

# TODO(ben): here is the other critical piece.  the goal here is to extract the
# model's idea of updated state, and SANITIZE IT before committing.  better
# to be conservative here.  if we skip an update, so be it.


def parse_state_update(block, state):

    # this is just for now, until we debug
    return state

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue

        m = UPDATE_RE.match(line)
        if not m:
            continue  # ignore malformed lines

        char, field, subkey, value = m.groups()
        char = char.upper()

        c = state.characters.setdefault(
            char,
            {
                "speech": [],
                "thoughts": [],
                "goal": None,
                "mood": "neutral",
                "relationships": {},
            },
        )

        value = value.strip()

        if field == "relationship" and subkey:
            c["relationships"][subkey.upper()] = value
        elif field in ["goal", "mood"]:
            c[field] = value

    return state


def extract_state_update(text):
    start = text.find("<STATE_UPDATE>")
    end = text.find("</STATE_UPDATE>")

    if start == -1 or end == -1:
        return None

    return text[start + len("<STATE_UPDATE>") : end].strip()

# get next contribution from one of the LLM participants (possibly LLM's 'silent'
# 'narrator' contribution).  This is intent only, so it is done with low temp
# and should be brief.  this is serialized w.r.t. further steps in the loop
# and so contributes directly to user-perceived TTFT.
def get_npc_intent() -> :

    while True:
        render_npc_eintent_query()
        scrub()
        if ok :
            break
        
    

if __name__ == "__main__":
    main()

# some notes on modern best practices:
# 1. hints everywhere
# 2. @dataclass
#    sub-point: immutable where possible (frozen=True)
# 3. pyright in strict mode (strict = true)
# 4. ruff/black
# 5. @property (sparingly, for validation); _underscore convention for encapsulation
# best approximation to encapsulation seems to be just to use underscores.

# sudo apt install mypy: static typecheck
# ... pylint: really thorough; people complain that it complains too much
# ... flake8
# The Immutable Types
#   Numbers: int, float, complex, bool (Yes, True and False are just special integers).
#   Sequences: str, tuple, range, and bytes.
#   Sets: frozenset (This is the immutable version of a set).
#   Special: NoneType (None is a singleton and never changes).

# type CharacterRef = tuple[int, int]  # we decided not to use this!  it's an int now
# or even 'from typing import NewType' and use NewType instead of type above.  stricter.
# to ensure dictionaries always have specific keys:
#    from typing import TypedDict
#    
#    class UserProfile(TypedDict):
#        username: str
#        age: int
#        is_active: bool
#    
#    # The type checker will now ensure you include these specific keys
#    user: UserProfile = {
#        "username": "coder123",
#        "age": 25,
#        "is_active": True
#    }

# note on packaging: if the user doesn't have the right C compiler for llama-cpp-python, they
# will get some cryptic error message about "Failed building wheel".  solution is to forewarn
# the user and point to a pre-build wheel
#
# if mypy complains that a third-party library like llama-cpp-python doesn't specify types,
# we should create "stubs" (.pyi files).  it works something like this:
# careate a 'typings' directory in the project
# in that dir create a file llama_cpp.pyi, and add the type signatures needed, like:
# llama_cpp.pyi
#    class Llama:
#        def __init__(self, model_path: str, n_ctx: int = 512): ...
#        def __call__(self, prompt: str) -> dict: ...
# then point mypy to it:
#    [tool.mypy]
#    mypy_path = "typings"


# cases:

# M->H
# we have an intent.  the intent has been scrubbed.
# we have the flowery prose, as this wsa streamed in n-1
# we need to compute all state updates
# we need to check the state updates
# we need to apply the state updates

# H->H:
# we do not have an intent and won't have one
# we have the flowery prose; it was input in n-1
# we need to compute all state updates
# we need to check the state updates
# we need to apply the state updates


# H->M
# we have the complete flowry prose
# there is not and won't be an intent

# M -> M
# we have the flowery prose
# there is an intent
# we need to compute all state updates
# we need to check the state updates
# we need to apply the state updates

# at the end of every turn, whetehr human or model, we have flowery prose.  we do NOT
# have updated state.  the human player doesn't need n-1's updated state.  model player
# does, so we calculate that first.
