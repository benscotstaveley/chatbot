from logging import getLogger
from config import Config
from llm_manager import LlmManager
from character import Character, choose_speaker, SpeakerRole, initialize_speaking_order, format_wts, update_speaking_order_for_delay

def loop(config:Config, llm_manager:LlmManager, roster:{Character}):

    logger = getLogger(__name__)
    logger.info("starting chat loop")

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
                user_input = input("\nUSER: ")
                messages.append({"role": "user", "content": user_input})

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

            case SpeakerRole.NPC:
                reply = llm_manager.generate_chat_reply(messages=messages, config=config, stream=stream)
                if not stream:
                    print(reply)
                messages.append({"role": "assistant", "content": reply})

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
