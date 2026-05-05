import sys
import os
import re
from llama_cpp import Llama, ChatCompletionRequestMessage
from chatloop import parser
from chatloop.State import State

UPDATE_RE = re.compile(
    r'^\s*([A-Za-z0-9_]+)\.(\w+)(?:\[(\w+)\])?\s*=\s*(.+)$'
)

MODEL_ROOT = "/mnt/models_nvme/models"

#MODEL_PATH=f"{MODEL_ROOT}/Mistral-Small-24B-ArliAI-RPMax-v1.4.Q5_K_M.gguf"
#MODEL_PATH=f"{MODEL_ROOT}/dolphin-2.6-mistral-7b.Q6_K.gguf"
#MODEL_PATH=f"{MODEL_ROOT}/Qwen2.5-32b-RP-Ink-Q4_K_M.gguf"
#MODEL_PATH=f"{MODEL_ROOT}/Mistral-rp-24b-karcher.i1-Q5_K_M.gguf"  # does not work: needs newer version of llama-cpp-python
#MODEL_PATH=f"{MODEL_ROOT}/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-IQ4_XS.gguf"
MODEL_PATH=f"{MODEL_ROOT}/cognitivecomputations_Dolphin-Mistral-24B-Venice-Edition-Q4_K_M.gguf"
system_behavior_prompt_file = "./sys_behavior.txt"
system_formatting_prompt_file = "./sys_formatting.txt"
initial_prompt_file = "../init.txt"
temperature=0  # for debug
context_size=24578
ngl=24

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

def main():
    # 1. SETUP: read prompts; Initialize the model

    # TODO: we may want to make some of this machine-generated, like
    # character cards.  we may want to pass a list of percentages
    # into the LLM to create a distilled personality

    with open(system_behavior_prompt_file, 'r') as f:
        system_behavior_prompt = " ".join(f.read().splitlines())

    with open(system_formatting_prompt_file, 'r') as f:
        system_formatting_prompt = " ".join(f.read().splitlines())

    with open(initial_prompt_file, 'r') as f:
        initial_prompt = " ".join(f.read().splitlines())

    state = State()

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
        {"role": "system", "content": system_behavior_prompt + system_formatting_prompt},
        {"role": "user", "content": initial_prompt}
    ]

    # new loop:
    while True:

        # choose_next_speaker().  # On first iteration it will be forced to first human.
        # need to force same speaker as created any dialog that is popped when the user uses that feature.

        # at the top of the loop we have n-1's unprocessed human or NPC output.
        # record what happens for HHMMH to cover all transitions.
        
        #if speaker_is_human(current_speaker):
            #   # human never needs to compute n-1's deltas b/c human sees the flowery speech and will compute it himself
            #   2 threads in parallel:
            #     Thread A (main program thread)
            #         do:
            #           get from stdin (or, on first time through the loop, from initial prompt)
            #           break if no special command
            #         canonicalize_human_output()  # this function correlated with format convenient for humans to output
            #     Thread B (spawn): compute and record pass n-1's authoritative state
            #         render_state_to_detailed_deltas()
            #         get_detailed_deltas() query the model for a *detailed* state delta output of all NPC activity
            #           since last call to this function.
            #           (this is all Relationship/Belief/CharacterState info.  it is a careful computation.  Note it can
            #            even include the prose of all the NPCs since the last human call)
            #         record_authoritative_history()
            #         if context_compression_required()
            #           compress_context()
            #     Wait for thread B to complete and terminate it

        #else:
            # note we could be computing quick deltas from human or NPC output
            # render_state_to_quick_deltas
            # get_quick_deltas().  (later overwrite with the detailed update that happens in parallel with human output
            #   this is deltas that affect the chosen NPC only)
            # record_quick_history()
            # get_npc_intent(): get next contribution from one of the LLM participants (possibly LLM's 'silent'
            #   'narrator' contribution).  This is intent only, so it is done with low temp
            #   and should be brief.  this is serialized w.r.t. further steps in the loop
            #   and so contributes directly to user-perceived TTFT.
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

        print("---------about to call create_chat_completion() with the following:---------")
        print(messages)
        print("----------------------------------------------------------------------------")

        # we would like to have this hook for debug, but it is unavailable in
        # our version of the library
        #print("----and here is the formatted prompt that will be passed to the model:------")
        #print(llm._format_chat_prompt(messages))
        #print("----------------------------------------------------------------------------")

    
        full_reply = ""
        for chunk in llm.create_chat_completion(
                stream=True,
                messages=messages,
                temperature=temperature,
                max_tokens=500
        ):

            delta = chunk["choices"][0]["delta"]
            finish_reason = chunk["choices"][0].get("finish_reason", None)

            if "content" in delta:
                text = delta["content"]
                print(text, end="", flush=True)
                full_reply += text

            if finish_reason in ["length"]:
                # TODO: we want to check for runaway text, and if that's not the
                # case, continue fetching text.  But this doesn't seem to work:
                # the next chunk comes back NULL and terminates the for loop
                print("\ngot 'length' termination; continue fetching response text\n")
            if finish_reason in ["stop"]:
                # TODO: handle these cases better.  for now, break the loop
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
        # TODO: also, discard last user input, if that's head of queue

        # 4. Append to history
        state.narration.append("USER: " + user_input)
        state.transcript.append(full_reply)
    
        messages.append({"role": "assistant", "content": full_reply})
        messages.append({"role": "user", "content": user_input})

# end of main()

# TODO: here is one of the two critical pieces.  we need a good function to
# create the 'messages' to send to the model from the state.
def render_message_list(state: State) -> [ChatCompletionRequestMessage] :
    lines = []

    # 1. Recent transcript (most important)
    for entry in state.transcript[-20:]:
        lines.append(entry)

    # 2. Optional structured summaries
    for name, char in state.characters.items():
        if char.get("goal"):
            lines.append(f"<INFO> {name} goal: {char['goal']}")

    return "\n".join(lines)

# TODO: here is the other critical piece.  the goal here is to extract the
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

        c = state.characters.setdefault(char, {
            "speech": [],
            "thoughts": [],
            "goal": None,
            "mood": "neutral",
            "relationships": {}
        })

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

    return text[start + len("<STATE_UPDATE>"):end].strip()


if __name__ == "__main__":
    main()

