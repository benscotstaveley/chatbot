
import re
from chatloop.State import State

TAG_RE = re.compile(r'^\s*(\([A-Za-z0-9_]+\)|\[([A-Za-z0-9_]+)\]|<([A-Za-z0-9_]+)>)\s*(.*)$')

def parse_llm_output(text: str, state: State):
    """
    Parses model output into structured state updates.
    Never fails; always degrades gracefully.
    """

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = TAG_RE.match(line)

        # 1. If line is well-formed
        if match:
            tag_full, bracket_tag, angle_tag, content = match.groups()

            tag = (bracket_tag or angle_tag or tag_full or "").strip("()<>[]")

            # Normalize tag
            tag = tag.upper()

            # Speech: [NAME]
            if tag.isalpha():
                state.characters.setdefault(tag, {
                    "speech": [],
                    "thoughts": [],
                    "goal": None,
                    "mood": "neutral"
                })
                state.characters[tag]["speech"].append(content)
                continue

            # Thought: (NAME_THOUGHT)
            if tag.endswith("_THOUGHT"):
                char = tag.replace("_THOUGHT", "")
                state.characters.setdefault(char, {
                    "speech": [],
                    "thoughts": [],
                    "goal": None,
                    "mood": "neutral"
                })
                state.characters[char]["thoughts"].append(content)
                continue

            # Narration
            if tag in ["NARRATION", "SYS", "WORLD"]:
                state.narration.append(content)
                continue

        # 2. Fallback: untagged text → narration (CRITICAL for robustness)
        state.narration.append(line)

    return state

