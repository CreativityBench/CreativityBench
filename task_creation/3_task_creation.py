import os, json, random, re
from utils import batch_call, load_json, save_json

MAX_CASES_PER_TIER = 2
ENTITY_COUNTS = [3, 6, 9, 12]
DIFFICULTIES = ["difficult", "medium", "low"]

OUTPUT = "outputs"
COMPARISONS_FILE = "2_comparisons.json"
TASKS_FILE = "3_tasks_ne.json"


TASK_PROMPT = """Your task is to generate three fields for a benchmark task.

## Gold
Entity: {gold_entity} | Part: {gold_part}
Gold Affordance:
{gold_affordance}

## Benchmark Task
{task}
Task Recipient: {recipient} (Condition: {recipient_condition})

## Entities in the Scene
{entities_desc}

---

Here are the instructions for the three fields you need to generate:
- **items**: List things referenced in the gold affordance's use_condition, environment_condition, and recipient_condition, plus the recipient itself. Describe each with the traits implied by those conditions. Also add 2-3 natural noise objects that fit the {scenario} scenario but are unrelated to the task (interactable=No). Do NOT add things similar to the entities already in the scene.
- **environment**: First-person scene description starting with "I am in the {scenario}. Around me there is ..." — naturally mention every entity and key item. You should not mention the index of the item. Please only mention the item's base name.
- **solution**: A structured dict with exactly four keys describing how to solve the task using {gold_entity}'s {gold_part}. Each value is a string in the format "... (Note: ...)" where the Note contains specific things a judge should explicitly verify when scoring. If no specific condition was explicitly needed, please directly write "(Note: NA)".

The four keys and what to write for each solution:
- "prepare_recipient": How to prepare the recipient to be applied for the task.
- "prepare_use_condition": How to set up the tool (gold part) to meet use_condition.
- "prepare_environment_condition": What environmental setup is needed to meet environment_condition.
- "apply_affordance": The core action about how to apply {gold_part}'s affordance to the recipient to solve the task. This is the most important step. Be comprehensive and detailed, referencing the specific physical/state attributes of the part that make this work.

Rules:
- Please be strictly grounded in the given gold affordance annotation and the part's listed attributes. Do not invent steps or attributes beyond what is annotated.
- Each step should be detailed and concrete, and at the same time do not be overly complex, and keep your sentence simple and straightforward.
- You should be aware that the Note in each key of the solution field is actually a judging reference.

Output ONLY valid JSON:
```json
{{
  "items": [
    {{
      "name": "The name of the additional items in the scene. Follow the instructions above to include items from conditions and noise items.",
      "description": "A concise description of the item, including traits implied by conditions. For noise items, describe their typical use or characteristics.",
      "interactable": "Yes or No. Give No only for the noise items that are unrelated to the task and should not be interacted with. Otherwise, give Yes."
    }}
  ],
  "environment": "A natural and coherent description of the scene, starting with 'I am in the {scenario}. Around me there is ...'. Make sure to mention every entity and key item in the scene naturally without mentioning their index.",
  "solution": {{
    "prepare_recipient": "A concise and clear description of how to prepare the recipient for the task according to the given gold affordance (Note: Critical judging reference of recipient_condition that need additional verification besides that in the task description, or give NA if no additional verification is needed.)",
    "prepare_use_condition": "A concise and clear description of how to set up the tool (gold part) to meet the use condition for solving the task (Note: Critical judging reference of use_condition that needed to be set up before the use its certain affordance, or give NA if no additional verification is needed.)",
    "prepare_environment_condition": "A concise and clear description of how to set up the environment to meet the environment condition for solving the task (Note: Critical judging reference of environment_condition that needed to be set up before applying the affordance to solve the target task, or give NA if no additional verification is needed.)",
    "apply_affordance": "A detailed description and comprehensive explanation of how to apply the affordance of the gold part to the recipient to solve the task (Note: Critical judging reference of the key attributes and affordance mechanism to verify before applying the affordance.)"
  }}
}}
```"""


def try_int(val, default=10):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def display_name(e_uid):
    """Strip the 'scenario+' prefix from an entity UID, keeping only the entity name."""
    return e_uid.split("+", 1)[-1]


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def entity_label(parts):
    """
    Return:
      - "similar" for entities that can show similar affordance (safe substitution style),
      - "not_similar" for entities that do not show similar affordance,
      - None for entities that should be excluded.
    """
    if not parts:
        return None
    if any(p.get("similar_affordance") == "Yes" and p.get("gold_change") == "Yes" for p in parts):
        return None
    if any(p.get("similar_affordance") == "Yes" and try_int(p.get("decision_making_difficulty"), 0) >= 3 for p in parts):
        return None
    dissimilar = all(
        p.get("similar_affordance") == "No" for p in parts
    )
    return "not_similar" if dissimilar else "similar"


def split_entities(entity_comparisons):
    similar, not_similar = [], []
    for e in entity_comparisons:
        parts = [p for p in e["parts"] if p is not None]
        label = entity_label(parts)
        if label == "similar":
            similar.append({"entity_uid": e["entity_uid"], "parts": parts})
        elif label == "not_similar":
            not_similar.append({"entity_uid": e["entity_uid"], "parts": parts})
    return similar, not_similar


def sample_by_difficulty(similar_entities, not_similar_entities, entity_count, difficulty, n_cases):
    if entity_count <= 0:
        return []

    sampled_sets = []
    seen = set()
    attempts = 0
    max_attempts = max(50, n_cases * 50)

    while len(sampled_sets) < n_cases and attempts < max_attempts:
        attempts += 1
        chosen = None
        if difficulty == "difficult":
            if len(similar_entities) < entity_count:
                break
            chosen = random.sample(similar_entities, entity_count)
        elif difficulty == "low":
            if len(not_similar_entities) < entity_count:
                break
            chosen = random.sample(not_similar_entities, entity_count)
        else:  # medium
            if entity_count % 2 != 0:
                break
            half = entity_count // 2
            if len(similar_entities) < half or len(not_similar_entities) < half:
                break
            chosen = random.sample(similar_entities, half) + random.sample(not_similar_entities, half)
            random.shuffle(chosen)

        key = tuple(sorted(e["entity_uid"] for e in chosen))
        if key in seen:
            continue
        seen.add(key)
        sampled_sets.append(chosen)

    return sampled_sets


def build_entity_description(e_uid, entity_lookup, gold_part=None):
    """Construct a comprehensive entity description from structured part attributes."""
    e_data = entity_lookup.get(e_uid, {})
    segments = []
    for p in e_data.get("parts", []):
        phys = p.get("physical_attributes", {}).get("summary", "")
        state = p.get("state_attributes", {}).get("summary", "")
        tag = " [gold part]" if p["part_name"] == gold_part else ""
        segments.append(f"{p['part_name']}{tag}: physical — {phys}; state — {state}.")
    return " ".join(segments) if segments else e_uid


def build_entities_desc(gold_e_uid, sampled_entities, entity_lookup):
    """Short summary for the LLM prompt (context only, not the output description)."""
    lines = []
    for e_uid, is_gold in [(gold_e_uid, True)] + [(e["entity_uid"], False) for e in sampled_entities]:
        e_data = entity_lookup.get(e_uid, {})
        parts_str = "; ".join(
            f"{p['part_name']}: {p.get('physical_attributes', {}).get('summary', '')} | {p.get('state_attributes', {}).get('summary', '')}"
            for p in e_data.get("parts", [])
        )
        prefix = "[GOLD] " if is_gold else ""
        lines.append(f"{prefix}{display_name(e_uid)}  —  {parts_str}")
    return "\n".join(lines)


def build_entity_judge(comp, gold_e_uid):
    entity_judge = {}
    for pc in comp["part_comparisons"]:
        entity_judge.setdefault(pc["entity_uid"], {})[pc["part_name"]] = pc["judge_output"]
    entity_judge[gold_e_uid] = {
        pc["part_name"]: pc["judge_output"]
        for pc in comp.get("gold_self_comparisons", [])
    }
    return entity_judge


def format_affordance_for_prompt(affordance):
    aff = dict(affordance)
    if "attribute" in aff and isinstance(aff["attribute"], list):
        aff["enabling_attributes"] = ", ".join(
            item[0] for item in aff["attribute"]
            if isinstance(item, list) and item
        )
        del aff["attribute"]
    return json.dumps(aff, indent=2)


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    entity_lookup = load_json(f"{OUTPUT}/1_entity_lookup.json")
    comparisons = load_json(f"{OUTPUT}/{COMPARISONS_FILE}")

    tasks_path = f"{OUTPUT}/{TASKS_FILE}"
    tasks = load_json(tasks_path) if os.path.exists(tasks_path) else []
    done_task_ids = {t.get("task_id") for t in tasks if t.get("task_id")}
    print(f"Resuming: {len(tasks)} tasks already done.")

    pending_prompts = []
    pending_prompt_keys = {}
    pending_meta = []

    for comp in comparisons:
        entity_parts = {}
        for pc in comp["part_comparisons"]:
            entity_parts.setdefault(pc["entity_uid"], []).append(pc["judge_output"])
        entity_comparisons = [{"entity_uid": uid, "parts": parts} for uid, parts in entity_parts.items()]

        similar_entities, not_similar_entities = split_entities(entity_comparisons)
        if not similar_entities and not not_similar_entities:
            continue

        scenario = comp["scenario"]
        gold_info = comp["gold_info"]
        gold_e_uid = gold_info["entity_uid"]
        gold_part = gold_info["part_name"]
        base_uid = f"{slugify(comp.get('gold_uid', f'{gold_e_uid}-{gold_part}'))}-iter{comp.get('iteration', 0)}"

        prompt_seed_entities = similar_entities + not_similar_entities
        prompt_n = min(max(ENTITY_COUNTS), len(prompt_seed_entities))
        llm_context_entities = random.sample(prompt_seed_entities, prompt_n) if prompt_n > 0 else []

        prompt = TASK_PROMPT.format(
            gold_entity=display_name(gold_e_uid),
            gold_part=gold_part,
            gold_affordance=format_affordance_for_prompt(gold_info["affordance"]),
            task=comp["task"].get("task", ""),
            recipient=comp["task"].get("recipient", ""),
            recipient_condition=comp["task"].get("recipient_condition", ""),
            entities_desc=build_entities_desc(gold_e_uid, llm_context_entities, entity_lookup),
            scenario=scenario,
        )
        prompt_key = f"{base_uid}"
        if prompt_key not in pending_prompt_keys:
            pending_prompt_keys[prompt_key] = len(pending_prompts)
            pending_prompts.append(prompt)
        llm_idx = pending_prompt_keys[prompt_key]

        entity_judge = build_entity_judge(comp, gold_e_uid)
        golds = [{
            "gold_entity": display_name(gold_e_uid),
            "gold_part": gold_part,
            "gold_affordance": gold_info["affordance"],
        }]

        for difficulty in DIFFICULTIES:
            for entity_count in ENTITY_COUNTS:
                sampled_sets = sample_by_difficulty(
                    similar_entities, not_similar_entities, entity_count, difficulty, MAX_CASES_PER_TIER
                )
                for sample_i, sampled_entities in enumerate(sampled_sets, start=1):
                    task_id = f"{base_uid}-{difficulty}-{entity_count}entity-sample{sample_i}"
                    if task_id in done_task_ids:
                        continue

                    entity_uids = [gold_e_uid] + [e["entity_uid"] for e in sampled_entities]
                    entities = [
                        {
                            "name": display_name(e_uid),
                            "description": build_entity_description(
                                e_uid, entity_lookup, gold_part=(gold_part if e_uid == gold_e_uid else None)
                            ),
                            "judge_output": entity_judge.get(e_uid, {}),
                        }
                        for e_uid in entity_uids
                    ]

                    setting = {
                        "difficulty": difficulty,
                        "entity_count": entity_count,
                        "sample_index": sample_i,
                        "max_cases_per_tier": MAX_CASES_PER_TIER,
                        "entity_counts_macro": ENTITY_COUNTS,
                        "scenario": scenario,
                        "level": comp.get("level"),
                        "cluster_size_range": comp.get("cluster_size_range"),
                        "iteration": comp.get("iteration"),
                        "cluster_id": comp.get("cluster_id"),
                    }

                    pending_meta.append({
                        "task_id": task_id,
                        "scenario": scenario,
                        "golds": golds,
                        "entities": entities,
                        "setting": setting,
                        "sampled_entity_count": entity_count,
                        "comp": comp,
                        "llm_idx": llm_idx,
                    })

    print(f"Pending: {len(pending_meta)} tasks to generate from {len(pending_prompts)} shared LLM calls.")

    if not pending_meta:
        print("Nothing to do.")
        print(f"\nDone! {len(tasks)} tasks → {tasks_path}")
        return

    llm_results = batch_call(pending_prompts, max_workers=1024, temperature=0.3)

    for meta in pending_meta:
        llm_result = llm_results[meta["llm_idx"]] if meta["llm_idx"] < len(llm_results) else None
        if not llm_result:
            print(f"  LLM failed for {meta['task_id']}")
            continue

        task_obj = {
            "task_id": meta["task_id"],
            "scenario": meta["scenario"],
            "setting": meta["setting"],
            "sampled_entity_count": meta["sampled_entity_count"],
            "golds": meta["golds"],
            "entities": meta["entities"],
            "items": llm_result.get("items", []),
            "environment": llm_result.get("environment", ""),
            "task": meta["comp"]["task"].get("task", ""),
            "solution": llm_result.get("solution", {}),
        }

        tasks.append(task_obj)
        print(f"  Created task {meta['task_id']}")

    save_json(tasks, tasks_path)
    print(f"\nDone! Created/updated {len(tasks)} tasks → {tasks_path}")


if __name__ == "__main__":
    main()
