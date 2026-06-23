from logging import getLogger
from config import Config
from llm_manager import LlmManager
from character import Character, choose_speaker, SpeakerRole, initialize_speaking_order, format_wts, update_speaking_order_for_delay
import copy
import re

def loop(config:Config, llm_manager:LlmManager, roster:{Character}):

    logger = getLogger(__name__)
    logger.info("starting chat loop")

    max_validate_attempts=3

    messages = [
        {"role": "system", "content": config.system_behavior_prompt + config.system_formatting_prompt},
    ]

    logger.info("messages list after adding system prompt:" + repr(messages))

    stream:bool = True

    initialize_speaking_order(roster)
    turn_number:int = 0
    prior_speaker:SpeakerRole = None

    while True:

        logger.debug("\n\ntop of loop.  WTS:" + format_wts(roster))
        this_speaker: Character = choose_speaker(roster)
        logger.debug("iterate with this character:" + repr(this_speaker))

        match this_speaker.role:
            case SpeakerRole.HUMAN:
                user_input = input(f"\n[{this_speaker.name}]: ")
                messages.append({"role": "user", "content": f"[{this_speaker.name}] " + user_input})

                logger.debug("-----messages-----" + repr(messages) + "------------------")

                # # construct a prompt specific for brief query
                # messages_iter = copy.deepcopy(messages)
                # messages_iter[-1]["content"] += " Be brief."
                # reply_brief = llm_manager.generate_chat_reply(messages_iter, config)
                # logger.debug("-------brief reply:-------\n" + reply_brief)

                # # construct a prompt specific for detailed query
                # messages_iter = copy.deepcopy(messages)
                # messages_iter[-1]["content"] += " Describe in detail."
                # reply_detailed = llm_manager.generate_chat_reply(messages_iter, config)
                # logger.debug("-------detailed reply:-------\n" + reply_detailed)

                # invariant: upon completion of Human role, 

            case SpeakerRole.NPC:

                # note that DIRECTOR is the name of a label that can appear in 'user' roles and is
                # explained early in the system prompt as an authoritative director of what the LLM
                # is to do now.

                # at this point, the last entry in 'messages' could have a user or assistant role.
                # NPC-after-NPC is the most common case where we get here with an assistant role at
                # the end of 'messages'; getting here after a HUMAN has spoken in the prior turn is
                # an example of a user role at the end of 'messages'.  Since this turn is NPC, we
                # start with adding a '[DIRECTOR]' tag asking the LLM for the very brief indication
                # of intent as to what this turn's NPC will do and/or say.  Note that [DIRECTOR] is
                # always classified as role:user.  Therefore in order to maintain strict
                # user/assistant alternation, if the last entry in 'messages' is role:assistant we
                # simply append a user entry to the 'messages' list.  But if the last entry is
                # already role:user, we do not append a new entry, but instead append the [DIRECTOR]
                # instruction to the end of the content field of the last entry.  Whatever we do, we
                # record it so we can unwind it.

                prompt_intent = f"[DIRECTOR] Considering the goals and most recent emotional state" \
                                 f" of {this_speaker.name}, provide a concise (under approximately 10 words) description of what" \
                                 f" {this_speaker.name} will say and do right now.  Don't provide actual dialog, just an outline:" \
                                 f" topics to raise, topics to avoid, mood and disposition, and specific actions"
                bookmark_turn = len(messages)
                if messages[-1]['role'] == "user":
                    messages[-1]['content'] += "\n" + prompt_intent
                else:
                    messages.append({"role": "user", "content": prompt_intent})

                # eventually: add an enhancement where we pre-populate the assistant turn with
                # "[NAME] ".  this will require some jinja magic.  it needs to be done
                # post-render(), so in llm_manager.generate_chat_reply().
                postrender_text = ""

                # at this point we are in a part of 'messages' that will be rolled back.
                # get a statement of intent suitable for verifying.

                # validation loop
                bookmark_validate = len(messages)
                validate_attempts=max_validate_attempts
                zero_temp_config = copy.copy(config)  # shallow copy is OK as we change only temp
                zero_temp_config.temp = 0.0
                while True :  # breaks on validate pass or 'max_validate_attempts' failures

                    logger.debug(f"validation attempt countdown: {validate_attempts}")
                    # at this point, either from first iteration or from end of prior iteration,
                    # we have 'messages' ending with user: [DIRECTOR].

                    logger.debug(f"calling generate_chat_reply() with [DIRECTOR] requesting concise summary intent: {messages}")
                    reply = llm_manager.generate_chat_reply(messages=messages, config=zero_temp_config, stream=False) # TODO(ben) pass postrender_text
                    logger.debug(f"got this concise statement of intent: {reply}")
                    best_statement_of_intent = reply

                    # this becomes the model's (ephemeral) statement of intent, initial attempt
                    messages.append({"role":"assistant", "content": postrender_text + reply})

                    # validate
                    messages.append({"role":"user",
                                    "content":f"[DIRECTOR] Consider the above statement of what {this_speaker.name} intends to say and do." \
                                     f" Does it directly contradict any of the goals, beliefs, and current mindset of {this_speaker.name}?" \
                                     f" Reply with one word: YES or NO"
                                    }
                                   )
                    logger.debug(f"calling generate_chat_reply with [DIRECTOR] asking for validation: {messages}")
                    reply = llm_manager.generate_chat_reply(messages=messages, config=zero_temp_config, stream=False)
                    logger.debug(f"got this answer of suitability: {reply}")

                    match reply:
                        case x if re.search(r"\bNO\b", x, re.IGNORECASE):
                            logger.debug("verification passed")
                            break
                        case x if re.search(r"\bYES\b", x, re.IGNORECASE):
                            logger.debug("verification failed")
                        case _:
                            logger.info(f"garbled response from LLM: {reply}")
                            logger.debug("verification failed")
                            # fall through to NO (fail) case

                    validate_attempts -= 1
                    if validate_attempts == 0:
                        break

                    # lay the groundwork for another attempt
                    messages.append({"role":"assistant", "content": "YES"})
                    messages.append({"role":"user", "content": "[DIRECTOR] State briefly (under approximately 10 words) why the intended action is unacceptable."})
                    reply = llm_manager.generate_chat_reply(messages=messages, config=zero_temp_config, stream=False)
                    messages.append({"role":"assistant", "content": reply})
                    logger.info(f"LLM rejected its statement of intent because: {reply}")
                    messages.append({"role":"user",
                                    "content":f"[DIRECTOR] Upon consideration, you found your previous statement of intent for {this_speaker.name}" \
                                    f" to be unacceptable." \
                                    f" Considering again the goals and most recent emotional state" \
                                    f" of {this_speaker.name}, provide a concise description of what" \
                                    f" {this_speaker.name} will say and do right now"
                                    }
                                   )


                # end of validation loop.  exit with validate_attempts>0: success, ==0: fail.
                # either way, best_statement_of_intent is as good as we're going to get.
                # TODO: should we do something in fail case other than log it?
                if validate_attempts==0:
                    logger.warning(f"validation repeatedly failed at turn {bookmark_turn}")
                del messages[bookmark_validate:]
                messages.append({"role":"assistant", "content": best_statement_of_intent})
                messages.append({"role":"user", "content": f"[DIRECTOR] Given the above statement of intent for {this_speaker.name}," \
                                f" write a complete response for {this_speaker.name} that embodies this intent and aligns with" \
                                f" the speaking style of {this_speaker.name} and stylistic requirements of the current situation."
                                })
                
                reply = llm_manager.generate_chat_reply(messages=messages, config=config, stream=stream)# TODO: preload with name of character
                if not stream:
                    print(reply)
                del messages[bookmark_turn:]
                messages.append({"role":"assistant", "content":reply})

            case SpeakerRole.NARRATOR:
                if turn_number==0:
                    narrator_speech = config.initial_prompt
                else:
                    narrator_speech = "**placeholder for narrator speech**"
                messages.append({"role": "user", "content": "[NARRATOR] " + narrator_speech})
                print("[NARRATOR] " + narrator_speech)

            case SpeakerRole.SILENT:
                if prior_speaker.role == SpeakerRole.SILENT:
                    print("(the silence stretches on...)")
                else:
                    print("(there is a pause in the conversation)")

            case _:
                raise ValueError(f"Unexpected role: this_speaker.role")

        if config.single_shot:
            break

        logger.debug(f"tokens: {len(llm_manager._cached_tokens)}")
        update_speaking_order_for_delay(roster)  # TODO(ben) include token count in calculation
        prior_speaker = this_speaker
        turn_number += 1
    # end of while forever
