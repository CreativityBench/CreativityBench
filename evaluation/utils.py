import os
import json
import re
import random
from models import call_model

MAX_TURNS = int(os.environ.get("MAX_TURNS", "15"))

SYS_STATIC = (
    "You are a creative problem-solver. Given a task, an environment description, "
    "detailed entity descriptions, and interactable items, identify the most creative "
    "and practical entity part that accomplishes the task."
)
SYS_COT = (
    "You are an expert at creative physical tool-use reasoning. "
    "Reason explicitly over entity parts, their physical properties, and affordances "
    "under task and environment constraints before choosing your final answer."
)
SYS_INTERACTIVE = (
    "You are a creative problem-solver exploring a scene step by step. "
    "You may inspect ONE thing in the scene at a time to learn its detailed description, "
    "or give your final answer when confident. Note that you have only {MAX_TURNS} turns to explore the scene and give your final answer."
).format(MAX_TURNS=MAX_TURNS)

_STATIC_ANSWER_FMT = (
    '{"gold_entity": "<entity name, case-sensitive and exact match>", "gold_part": "<part name, case-sensitive and exact match>", "how_to_use": "<detailed instructions>"}'
)
_COT_ANSWER_FMT = (
    "{\n"
    '  "reasoning": {\n'
    '    "task_goal": "...",\n'
    '    "success_condition": "...",\n'
    '    "identified_constraints": ["...", "..."],\n'
    '    "candidate_parts": [\n'
    "      {\n"
    '        "entity": "...",\n'
    '        "part": "...",\n'
    '        "inferred_physical_properties": ["...", "..."],\n'
    '        "affordances_for_task": ["...", "..."],\n'
    '        "constraint_check": "..." \n'
    "      }\n"
    "    ],\n"
    '    "reasoning_plan": [\n'
    "      {\n"
    '        "step": 1,\n'
    '        "action": "...",\n'
    '        "tool_part_used": "entity:part",\n'
    '        "affordance_used": "...",\n'
    '        "why_it_works": "..." \n'
    "      }\n"
    "    ],\n"
    '    "creative_reasoning_summary": "1-3 sentences on novelty + practicality"\n'
    "  },\n"
    '  "gold_entity": "<entity name, case-sensitive and exact match>",\n'
    '  "gold_part": "<part name, case-sensitive and exact match>",\n'
    '  "how_to_use": "<detailed instructions>"\n'
    "}"
)
_INTERACTIVE_EXPLORE_FMT = (
    '{"action": "explore", "target": "<exact name from list, case-sensitive and exact match>"}'
)
_INTERACTIVE_ANSWER_FMT = (
    '{"action": "answer", "gold_entity": "<entity name, case-sensitive and exact match>", "gold_part": "<part name, case-sensitive and exact match>", "how_to_use": "<detailed instructions>"}'
)

_FORMAT_REMINDER = (
    "Write your reasoning as plain text first, then end with a JSON object:\n\n"
    f"  1. To explore:  <reasoning>\n  {_INTERACTIVE_EXPLORE_FMT}\n\n"
    f"  2. To answer:   <reasoning>\n  {_INTERACTIVE_ANSWER_FMT}"
)
_PARSE_ERROR_MSG = (
    "I could not parse a valid JSON action from your response. "
    f"Please try again.\n{_FORMAT_REMINDER}"
)
_UNKNOWN_ACTION_MSG = (
    "Unknown action — please use \"explore\" or \"answer\".\n"
    f"{_FORMAT_REMINDER}"
)


# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_json(text: str):
    """Extract the first JSON object from a string, or return None."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def parse_response(text: str) -> tuple:
    """For static responses: split into (reasoning_text, answer_dict).

    The model is instructed to write reasoning as plain text first, then a
    JSON object on the final lines.  We find the first '{' that belongs to a
    parseable JSON block and treat everything before it as reasoning.

    Returns:
        (reasoning, answer_dict) — answer_dict is {} when no valid JSON found
        (in that case reasoning contains the full raw response).
    """
    text = text.strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        reasoning = text[:m.start()].strip()
        try:
            return reasoning, json.loads(m.group())
        except Exception:
            pass
    # No parseable JSON — put everything in reasoning, leave answer blank
    return text, {}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _clean(desc: str) -> str:
    """Strip [gold part] annotation so we don't leak the answer."""
    return desc.replace(" [gold part]", "").replace("[gold part] ", "")


def _fmt_entity(entity: dict) -> str:
    """Format one entity's description into a readable block."""
    desc = _clean(entity["description"])
    # Parts are separated by '.. ' followed by a lowercase part-name token
    parts = re.split(r'\.\.\s+(?=[a-z_]+:)', desc)
    lines = [f"\n[{entity['name']}]"]
    for part in parts:
        part = part.strip().rstrip('.')
        if part:
            lines.append(f"  • {part}")
    return "\n".join(lines)


# ── Static mode ────────────────────────────────────────────────────────────────

def build_static_prompt(task: dict) -> str:
    blocks = [
        f"TASK:\n{task['task']}\n",
        f"ENVIRONMENT:\n{task['environment']}\n",
        "=== ENTITIES AVAILABLE (full descriptions) ===",
    ]
    for entity in task["entities"]:
        blocks.append(_fmt_entity(entity))

    interactable = [it for it in task["items"] if it["interactable"] == "Yes"]
    if interactable:
        blocks.append("\n=== OTHER ITEMS IN SCENE ===\n")
        for it in interactable:
            blocks.append(f"  • {it['name']}: {it['description']}")

    blocks.append(
        "\nIdentify the best entity part to creatively accomplish the task. "
        "Please first provide your reasoning process, and then finally respond with "
        "a JSON object containing the three answer fields:\n"
        "<your reasoning process here>\n" + _STATIC_ANSWER_FMT
    )
    return "\n".join(blocks)


def build_cot_prompt(task: dict) -> str:
    blocks = [
        f"TASK:\n{task['task']}\n",
        f"ENVIRONMENT:\n{task['environment']}\n",
        "=== ENTITIES AVAILABLE (full descriptions) ===",
    ]
    for entity in task["entities"]:
        blocks.append(_fmt_entity(entity))

    interactable = [it for it in task["items"] if it["interactable"] == "Yes"]
    if interactable:
        blocks.append("\n=== OTHER ITEMS IN SCENE ===\n")
        for it in interactable:
            blocks.append(f"  • {it['name']}: {it['description']}")

    blocks.append(
        "\nRequired reasoning procedure:\n"
        "1. State the task goal and concrete success condition.\n"
        "2. List relevant entities/items from the scene (do not add new tools).\n"
        "3. For candidate parts, infer physical properties and affordances tied to this task.\n"
        "4. Build a minimal step-by-step reasoning plan grounded in those affordances.\n"
        "5. Validate the chosen part against task/environment constraints.\n"
        "6. Generate final answer fields strictly based on the intermediate reasoning.\n\n"
        "Return JSON only (no markdown) with this format:\n"
        + _COT_ANSWER_FMT
    )
    return "\n".join(blocks)


# ── Interactive mode ───────────────────────────────────────────────────────────

def _all_names_shuffled(task: dict) -> list:
    names = [e["name"] for e in task["entities"]]
    names += [it["name"] for it in task["items"] if it["interactable"] == "Yes"]
    random.shuffle(names)
    return names


def _find_description(task: dict, target: str):
    tl = target.lower()
    for e in task["entities"]:
        if e["name"].lower() == tl:
            return _clean(e["description"])
    for it in task["items"]:
        if it["name"].lower() == tl:
            return it["description"]
    return None


def _interactive_initial(task: dict) -> str:
    names = _all_names_shuffled(task)
    name_list = "\n".join(f"  - {n}" for n in names)
    return (
        f"TASK:\n{task['task']}\n\n"
        f"ENVIRONMENT:\n{task['environment']}\n\n"
        "=== THINGS IN THE SCENE ===\n"
        f"{name_list}\n\n"
        "INSTRUCTIONS:\n"
        "• We strongly encourage you to explore the things in the scene to make a well-reasoned decision before committing to a final answer.\n"
        "• You may only explore ONE thing per turn.\n"
        "• At every turn, first write your reasoning as plain text, then end with a JSON object.\n\n"
        "At each turn, respond in one of these two formats:\n\n"
        f"1. To explore a thing in the scene:\n<your reasoning here>\n{_INTERACTIVE_EXPLORE_FMT}\n\n"
        f"2. To give your final answer:\n<your reasoning here>\n{_INTERACTIVE_ANSWER_FMT}"
    )


def _turns_suffix(turns_left: int) -> str:
    """Return a notice appended to every feedback message."""
    if turns_left == 1:
        return (
            "\n\n[1 turn remaining — this is your last turn. "
            "You MUST give your final answer now.]"
        )
    return f"\n\n[{turns_left} turns remaining.]"


def run_interactive(model_name: str, task: dict, temperature: float = 0.0) -> dict:
    """Run a full interactive session; return answer + raw history + conversation_log."""
    prompt = _interactive_initial(task)
    messages = [
        {"role": "system", "content": SYS_INTERACTIVE},
        {"role": "user",   "content": prompt},
    ]
    history          = []   # raw message log (assistant + user turns, flat)
    conversation_log = []   # structured per-round log (model output + feedback)
    last_reasoning, last_parsed = "", None

    for turn in range(MAX_TURNS):
        response = call_model(model_name, messages, temperature=temperature)
        messages.append({"role": "assistant", "content": response})

        # Split plain-text reasoning from the trailing JSON (same as static)
        reasoning, parsed = parse_response(response)
        history.append({"turn": turn + 1, "assistant": response, "reasoning": reasoning})

        turns_left = MAX_TURNS - (turn + 1)

        # Start a log entry for this round; feedback filled in below
        log_entry = {
            "round":          turn + 1,
            "model_reasoning": reasoning,
            "model_action":    parsed if parsed else None,
        }

        # ── Unparseable / missing action → ask model to correct format ─────
        if not parsed or "action" not in parsed:
            feedback = _PARSE_ERROR_MSG + _turns_suffix(turns_left)
            messages.append({"role": "user", "content": feedback})
            history.append({"turn": turn + 1, "user": feedback, "note": "parse_correction"})
            log_entry["feedback"] = feedback
            log_entry["note"]     = "parse_correction"
            conversation_log.append(log_entry)
            continue

        last_reasoning, last_parsed = reasoning, parsed

        # ── Final answer ───────────────────────────────────────────────────
        if parsed["action"] == "answer":
            conversation_log.append(log_entry)   # no feedback on final answer
            return {
                "answer": {
                    "reasoning":   reasoning,
                    "gold_entity": parsed.get("gold_entity", ""),
                    "gold_part":   parsed.get("gold_part",   ""),
                    "how_to_use":  parsed.get("how_to_use",  ""),
                },
                "turns": turn + 1,
                "conversation_log": conversation_log,
            }

        # ── Explore ────────────────────────────────────────────────────────
        if parsed["action"] == "explore":
            target = parsed.get("target", "")
            desc = _find_description(task, target)
            if desc:
                follow_up = (
                    f"=== DESCRIPTION OF: {target} ===\n{desc}\n\n"
                    "Continue: explore another item, or give your final answer."
                )
            else:
                follow_up = (
                    f"'{target}' was not found in the scene. "
                    "Choose an exact name from the original list and try again."
                )
            follow_up += _turns_suffix(turns_left)
            messages.append({"role": "user", "content": follow_up})
            history.append({"turn": turn + 1, "user": follow_up})
            log_entry["feedback"] = follow_up
            conversation_log.append(log_entry)
            continue

        # ── Unknown action → correction feedback ───────────────────────────
        feedback = _UNKNOWN_ACTION_MSG + _turns_suffix(turns_left)
        messages.append({"role": "user", "content": feedback})
        history.append({"turn": turn + 1, "user": feedback, "note": "action_correction"})
        log_entry["feedback"] = feedback
        log_entry["note"]     = "action_correction"
        conversation_log.append(log_entry)

    # Max-turns reached — return best available answer
    ans = {
        "reasoning":   last_reasoning,
        "gold_entity": last_parsed.get("gold_entity", "") if last_parsed else "",
        "gold_part":   last_parsed.get("gold_part",   "") if last_parsed else "",
        "how_to_use":  last_parsed.get("how_to_use",  "") if last_parsed else "",
    }
    
    return {
        "answer":           ans,
        "turns":            MAX_TURNS,
        "conversation_log": conversation_log,
    }
