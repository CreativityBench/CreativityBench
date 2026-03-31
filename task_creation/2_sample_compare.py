import os, re, json, random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from utils import call_gpt, batch_call, load_json, save_json

N_PER_SCENARIO = 10
CLUSTER_SIZE_RANGE = [(2, 4), (5, 10), (10, 50)]
LEVEL_RANGE = [0, 1, 2, 3, 4, 5]
N_ENTITIES = 30
N_SIMILAR = 15

OUTPUT = "outputs"
COMPARISONS_FILE = "2_comparisons.json"

SCENARIOS = [
    "kitchen", "living_room", "bedroom", "bathroom", "garage", "home_office", "dining_room", "garden"
]


TASK_PROMPT = """"You are writing a first-person benchmark question for a creative problem-solving task. Given a gold affordance, your job is to invent a concrete, realistic situation that the affordance can solve — without revealing the gold entity, part, or affordance mechanism. The task should also be aligned with the given scenario.

Given Scenario: {scenario}
Gold Entity: {entity_uid}
Gold Part: {part_name}
Gold Affordance:
{affordance}

**Your process:**
Step 1 — Pick a concrete recipient from `example_recipient` (or one that satisfies `recipient_condition`).
Step 2 — Invent a specific real-world situation that person encounters involving that recipient, conforming with the given scenario.
Step 3 — Describe the situation and goal in everyday language, then ask "What can I use?" or "What should I use and how?".

**Task Creation Rules:**
- Start from the SITUATION and RECIPIENT, not from the affordance mechanism. Ask yourself: "What real problem is this person facing?" not "What does the affordance do?"
- Be specific and concrete. Name the recipient object. Give context (where, what happened, what they want to achieve).
- Do NOT paraphrase or echo the affordance description. If the affordance is "dampen a surface", do NOT write "make a spot wet". Describe WHY the person needs it.
- Do NOT name or hint at the gold entity or part.
- Do NOT describe what needs to happen, but describe your goal/problem.

**Self-Check Requirements:**
- When creating the task, also evaluate whether the provided gold entity/part and its gold affordance are still appropriate and reasonable for solving it.
- Note that these gold affordances are intended for emergency or unusual use cases, so do not judge them based on how common they are. Instead, evaluate them based on factors including effectiveness, safety, and social acceptability.
- Always try to create a task that satisfies all of the requirements above, but at the same time, also be honest and objective in your self-checks.
- If you genuinely believe no valid task can satisfy all of these self-check conditions, you may write `No` in some fields. However, you should make your best effort to create a valid task whenever possible.

**Examples (affordance: "apply small amount of moisture to a surface"):**
✅ GOOD: "I am currently in the bathroom of my house. My fabric jacket has a small stubborn stain from yesterday's lunch. I want to treat just that spot before tossing it in the wash, but I don't have a spray bottle or wet cloth nearby. What can I use?"
  → Concrete: specific object (fabric jacket), specific problem (stain), specific constraint (no wet cloth). Affordance is not mentioned.

❌ BAD: "I'm trying to quickly put a small, localized damp spot onto an absorbent surface. What can I use?"
  → Describes the affordance mechanism directly. No real situation. Not grounded.

❌ BAD: "I need to use the sponge tip to apply moisture to my shirt." → Leaks entity and mechanism.

Output JSON:
```json
{{
  "task": "Concrete first-person scenario question grounded in a specific recipient and situation, hiding the gold entity/part/affordance",
  "recipient": "The specific object or thing receiving this affordance (pick from example_recipient or invent one matching recipient_condition)",
  "recipient_condition": "Required state or attribute of the recipient",
  "self_check_safety": "Yes or No, under your created task scenario, is it safe to use the given gold affordance of the gold entity/part to solve your task?",
  "self_check_acceptability": "Yes or No, under your created task scenario, are people still willing to use the given gold affordance of the gold entity/part to solve your task under emergency situation?",
  "self_check_effectiveness": "Yes or No, under your created task scenario, is it still reasonable and effective to use the given gold affordance of the gold entity/part to solve your task without side effects?"
  "self_check_reason": "Your reasoning for the above three self-checks. Be detailed and specific, reflecting on safety concerns, social acceptability, effectiveness, and potential side effects."
}}
```"""


COMPARE_PROMPT = """You are judging whether a target part can fulfill a similar role to a gold affordance for a given task.

## Gold
Entity: {gold_entity} | Part: {gold_part}
Physical Attributes of Part {gold_part}:
{gold_physical_attrs}
State Attributes of Part {gold_part}:
{gold_state_attrs}
Gold Affordance:
{gold_affordance}

## Target Part
Entity: {target_entity} | Part: {target_part}
Physical Attributes of Part {target_part}:
{physical_attrs}
State Attributes of Part {target_part}:
{state_attrs}
Existing Affordances of Part {target_part}:
{existing_affordances}

## Task
{task}
Recipient: {recipient}

---

**Step 1 — Similarity Judgment**
Determine whether this part ALONE (without any other part of the entity) can perform a functionally similar role to solve the task — i.e., it serves the same purpose as the gold affordance. Base your judgment strictly on the part's provided physical and state attributes.

**Step 2 — Affordance Annotation** (only if similar_affordance = Yes)
If an existing affordance already matches, adapt it. Otherwise write a new one. All fields must be grounded in the provided attributes. If similar_affordance = No, set ALL affordance fields to "NA".

Field definitions:
- use_condition: preparation needed to access this affordance; "NA" if part is free and directly usable (based on visibility/availability from state attributes)
- environment_condition: external environmental conditions required (not about the part or recipient); "NA" if none
- attribute: list of [["attribute statement", "physical/state"]], indicating all given target part attributes enabling this affordance;
- recipient_condition: required attributes of the recipient (shape, size, rigidity, material, etc.), fine-grained
- example_recipient: 3-4 concrete examples satisfying recipient_condition
- failure_case: all situations where this affordance fails — use condition failures, environment failures, recipient incompatibility, action/skill failures

**Step 3 — Gold Comparison** (only if similar_affordance = Yes)
Systematically compare gold vs. target across ALL of the following aspects. For each aspect, explicitly state which side is better and by how much:
- Accessibility and use/env conditions (how easy to access and activate)
- Effectiveness for the task (how well it achieves the goal, how directly it addresses the problem)
- Future consequences (irreversible damage, side effects, mess)
- How willing people are to use it this way (social acceptability, effort)
- How common this usage is in everyday life
- Safety and ethical considerations
After comparing all aspects, make a final decision of which one is better overall for solving the task, and reflect objectively on the trade-offs and whether the gold affordance should be changed or not.

**Step 4 — Decision Making Difficulty**
If most aspects clearly favor one side, gold_change is straightforward. If several aspects are roughly equal, or pros and cons are genuinely hard to weigh against each other, explicitly acknowledge that uncertainty and reflect it with a high decision_making_difficulty score.
- Decision making difficulty: 1 = very easy to decide whether to change or keep gold (one side clearly better across most aspects); 5 = extremely difficult (aspects are close or contradictory, genuinely uncertain which is better)
- Give NA only if similar_affordance is No. Otherwise please give an objective assessment of the decision making difficulty.
- A high score (4-5) is appropriate whenever multiple aspects are roughly tied or point in opposite directions. Do not default to low scores when the comparison is genuinely hard.

Output JSON:
```json
{{
  "similar_affordance_reason": "whether this part ALONE can show similar affordance to solve the task, based on which attributes and conditions",
  "similar_affordance": "Yes or No",
  "affordance": {{
    "use_condition": "...",
    "environment_condition": "...",
    "attribute": [["attribute statement", "physical/state"]],
    "affordance": "...",
    "recipient_condition": "...",
    "example_recipient": ["...", "...", "..."],
    "failure_case": "..."
  }},
  "gold_change_reason": "For each aspect — accessibility, effectiveness, future consequences, willingness, commonness, safety — state which side is better and why. Then give a final balanced verdict. If aspects conflict or are too close to call, say so explicitly.",
  "gold_change": "Yes or No",
  "decision_making_difficulty": "1-5"
}}
```"""


def try_int(val, default=10):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def gold_part_is_dominant(self_results):
    for r in self_results:
        if r is None:
            continue
        if r.get("gold_change") == "Yes":
            return False
        if r.get("similar_affordance") == "Yes" and try_int(r.get("decision_making_difficulty"), 0) >= 3:
            return False
    return True


def is_empty_value(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, dict):
        return not value or any(is_empty_value(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return not value or any(is_empty_value(v) for v in value)
    return False


def normalize_comparison_result(result):
    if not isinstance(result, dict):
        return None

    normalized = dict(result)
    similar_affordance = str(normalized.get("similar_affordance", "")).strip()
    normalized["similar_affordance"] = similar_affordance

    if similar_affordance == "No":
        reason = normalized.get("similar_affordance_reason")
        if is_empty_value(reason):
            return None
        normalized["affordance"] = "NA"
        normalized["gold_change_reason"] = reason
        normalized["gold_change"] = "No"
        normalized["decision_making_difficulty"] = "NA"
        return normalized

    if similar_affordance != "Yes":
        return None

    required_top_level_fields = [
        "similar_affordance_reason",
        "affordance",
        "gold_change_reason",
        "gold_change",
        "decision_making_difficulty",
    ]
    if any(is_empty_value(normalized.get(field)) for field in required_top_level_fields):
        return None

    if str(normalized["decision_making_difficulty"]).strip().upper() == "NA":
        return None

    affordance = normalized.get("affordance")
    if not isinstance(affordance, dict):
        return None

    required_affordance_fields = [
        "use_condition",
        "environment_condition",
        "attribute",
        "affordance",
        "recipient_condition",
        "example_recipient",
        "failure_case",
    ]
    if any(is_empty_value(affordance.get(field)) for field in required_affordance_fields):
        return None

    return normalized


def extract_level(level_str):
    if "Normal 0" in str(level_str):
        return 0
    for i in range(1, 6):
        if f"Emergency {i}" in str(level_str):
            return i
    return -1


def cosine_dist(a, b):
    a, b = np.array(a), np.array(b)
    return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def min_dist_to_gold(e_uid, entity_lookup, emb_dict, gold_emb):
    dists = []
    for part in entity_lookup[e_uid]["parts"]:
        for i in range(len(part["functional_affordances"])):
            uid = f"{e_uid}::{part['part_name']}::{i}"
            if uid in emb_dict:
                dists.append(cosine_dist(gold_emb, emb_dict[uid]))
    return min(dists) if dists else 1.0


def get_base_entity_name(entity_name):
    return re.sub(r"\s*\d+\s*$", "", entity_name).lower().strip()


def select_unique_base_entities(candidates, limit, entity_lookup, used_bases=None):
    used_bases = set() if used_bases is None else set(used_bases)
    selected = []
    for e_uid in candidates:
        base_name = get_base_entity_name(entity_lookup[e_uid]["entity_name"])
        if base_name in used_bases:
            continue
        selected.append(e_uid)
        used_bases.add(base_name)
        if len(selected) >= limit:
            break
    return selected, used_bases


def sample_gold(scen_clusters, aff_lookup, target_level, cluster_size_range, used_entity_uids=None):
    used_entity_uids = used_entity_uids or set()
    min_size, max_size = cluster_size_range
    valid = {k: v for k, v in scen_clusters.items() if min_size <= len(v) <= max_size}
    if not valid:
        return None, None
    for _ in range(200):
        cid = random.choice(list(valid.keys()))
        uid = random.choice(valid[cid])
        if aff_lookup[uid]["entity_uid"] in used_entity_uids:
            continue
        level = extract_level(aff_lookup[uid]["affordance"].get("level", ""))
        if level == target_level:
            return uid, cid
    return None, None


def sample_entities(scenario, gold_e_uid, entity_lookup, emb_dict, gold_emb):
    all_uids = [u for u, v in entity_lookup.items()
                if v["scenario"] == scenario and u != gold_e_uid]
    gold_base = get_base_entity_name(entity_lookup[gold_e_uid]["entity_name"])
    remaining = [u for u in all_uids
                 if get_base_entity_name(entity_lookup[u]["entity_name"]) != gold_base]
    by_dist = sorted(remaining, key=lambda u: min_dist_to_gold(u, entity_lookup, emb_dict, gold_emb))
    similar, used_bases = select_unique_base_entities(
        by_dist, N_SIMILAR, entity_lookup
    )
    n_dissimilar = max(0, N_ENTITIES - len(similar))
    dissimilar, _ = select_unique_base_entities(
        reversed(by_dist), n_dissimilar, entity_lookup, used_bases=used_bases
    )
    return similar + dissimilar


def format_affordance_for_prompt(affordance):
    aff = dict(affordance)
    if "attribute" in aff and isinstance(aff["attribute"], list):
        aff["enabling_attributes"] = ", ".join(
            item[0] for item in aff["attribute"]
            if isinstance(item, list) and item
        )
        del aff["attribute"]
    return json.dumps(aff, indent=2)


def format_affordance_list_for_prompt(affordances):
    return json.dumps(
        [json.loads(format_affordance_for_prompt(aff)) for aff in affordances],
        indent=2,
    )


def process_one_iteration(args):
    scenario, level, cluster_size_range, iteration, aff_lookup, entity_lookup, emb_dict, clusters, done_keys, results_lock, results_file = args
    
    scen_clusters = clusters.get(scenario, {})
    key = (scenario, level, tuple(cluster_size_range), iteration)
    if key in done_keys:
        return None
    
    used_gold_entities = {r["gold_info"]["entity_uid"]
                          for r in results_file if r.get("scenario") == scenario
                          and r.get("level") == level
                          and r.get("cluster_size_range") == list(cluster_size_range)}
    
    for retry in range(100):
        gold_uid, cluster_id = sample_gold(scen_clusters, aff_lookup, level, cluster_size_range, used_gold_entities)
        if gold_uid is None:
            return None
        
        gold_info = aff_lookup[gold_uid]
        gold_emb = emb_dict[gold_uid]
        
        _gold_part_data = next(
            (p for p in entity_lookup[gold_info["entity_uid"]]["parts"]
             if p["part_name"] == gold_info["part_name"]), {}
        )
        gold_physical_attrs = json.dumps(_gold_part_data.get("physical_attributes", {}), indent=2)
        gold_state_attrs = json.dumps(_gold_part_data.get("state_attributes", {}), indent=2)
        gold_affordance_str = format_affordance_for_prompt(gold_info["affordance"])
        
        task_result = call_gpt(TASK_PROMPT.format(
            scenario=scenario,
            entity_uid=gold_info["entity_uid"],
            part_name=gold_info["part_name"],
            affordance=gold_affordance_str,
        ), temperature=0.3)
        if not task_result:
            used_gold_entities.add(gold_info["entity_uid"])
            continue
        if task_result.get("self_check_effectiveness", "No") != "Yes" or task_result.get("self_check_safety", "No") != "Yes" or task_result.get("self_check_acceptability", "No") != "Yes":
            print("  Task self-check failed, trying next gold.")
            used_gold_entities.add(gold_info["entity_uid"])
            continue
        
        gold_entity_parts = entity_lookup[gold_info["entity_uid"]]["parts"]
        other_parts = [p for p in gold_entity_parts if p["part_name"] != gold_info["part_name"]]
        
        if other_parts:
            self_prompts = [COMPARE_PROMPT.format(
                gold_entity=gold_info["entity_uid"],
                gold_part=gold_info["part_name"],
                gold_physical_attrs=gold_physical_attrs,
                gold_state_attrs=gold_state_attrs,
                gold_affordance=gold_affordance_str,
                target_entity=gold_info["entity_uid"],
                target_part=part["part_name"],
                physical_attrs=json.dumps(part["physical_attributes"], indent=2),
                state_attrs=json.dumps(part["state_attributes"], indent=2),
                existing_affordances=format_affordance_list_for_prompt(part["functional_affordances"]),
                task=task_result.get("task", ""),
                recipient=task_result.get("recipient", ""),
            ) for part in other_parts]
            self_results = batch_call(self_prompts, max_workers=16, temperature=0.0)
            processed_self_pairs = [
                (part, normalized)
                for part, res in zip(other_parts, self_results)
                for normalized in [normalize_comparison_result(res)]
                if normalized is not None
            ]
            other_parts = [part for part, _ in processed_self_pairs]
            self_results = [res for _, res in processed_self_pairs]
        else:
            self_results = []

        if not gold_part_is_dominant(self_results):
            used_gold_entities.add(gold_info["entity_uid"])
            continue
        
        used_gold_entities.add(gold_info["entity_uid"])
        
        gold_self_comparisons = [
            {"part_name": part["part_name"], "judge_output": res}
            for part, res in zip(other_parts, self_results)
        ]
        
        sampled = sample_entities(scenario, gold_info["entity_uid"], entity_lookup, emb_dict, gold_emb)
        compare_tasks = [(e_uid, part) for e_uid in sampled for part in entity_lookup[e_uid]["parts"]]
        prompts = [COMPARE_PROMPT.format(
            gold_entity=gold_info["entity_uid"],
            gold_part=gold_info["part_name"],
            gold_physical_attrs=gold_physical_attrs,
            gold_state_attrs=gold_state_attrs,
            gold_affordance=gold_affordance_str,
            target_entity=e_uid,
            target_part=part["part_name"],
            physical_attrs=json.dumps(part["physical_attributes"], indent=2),
            state_attrs=json.dumps(part["state_attributes"], indent=2),
            existing_affordances=format_affordance_list_for_prompt(part["functional_affordances"]),
            task=task_result.get("task", ""),
            recipient=task_result.get("recipient", ""),
        ) for e_uid, part in compare_tasks]
        
        compare_results = batch_call(prompts, max_workers=256, temperature=0.0)
        processed_compare_pairs = [
            ((e_uid, part), normalized)
            for (e_uid, part), res in zip(compare_tasks, compare_results)
            for normalized in [normalize_comparison_result(res)]
            if normalized is not None
        ]
        
        part_comparisons = [
            {"entity_uid": e_uid, "part_name": part["part_name"], "judge_output": res}
            for (e_uid, part), res in processed_compare_pairs
        ]
        
        result = {
            "scenario": scenario,
            "level": level,
            "cluster_size_range": list(cluster_size_range),
            "iteration": iteration,
            "gold_uid": gold_uid,
            "gold_info": gold_info,
            "cluster_id": cluster_id,
            "cluster_size": len(scen_clusters[cluster_id]),
            "task": task_result,
            "sampled_entities": sampled,
            "gold_self_comparisons": gold_self_comparisons,
            "part_comparisons": part_comparisons,
        }
        
        with results_lock:
            results_file.append(result)
            save_json(results_file, f"{OUTPUT}/{COMPARISONS_FILE}")
        
        return result
    
    return None


def process_combination(args):
    scenario, level, cluster_size_range, aff_lookup, entity_lookup, emb_dict, clusters, done_keys, results_lock, results_file = args
    
    print(f"  [{scenario}] Level {level}, Cluster size {cluster_size_range}")
    
    for iteration in range(N_PER_SCENARIO):
        result = process_one_iteration((
            scenario, level, cluster_size_range, iteration,
            aff_lookup, entity_lookup, emb_dict, clusters,
            done_keys, results_lock, results_file
        ))
        if result:
            print(f"    Iteration {iteration + 1}/{N_PER_SCENARIO}: done")
        else:
            print(f"    Iteration {iteration + 1}/{N_PER_SCENARIO}: failed")


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    aff_lookup = load_json(f"{OUTPUT}/1_affordance_lookup.json")
    entity_lookup = load_json(f"{OUTPUT}/1_entity_lookup.json")
    clusters = load_json(f"{OUTPUT}/1_clusters.json")
    emb_dict = load_json(f"{OUTPUT}/1_embeddings.json")
    
    results_path = f"{OUTPUT}/{COMPARISONS_FILE}"
    results = load_json(results_path) if os.path.exists(results_path) else []
    done_keys = {(r["scenario"], r["level"], tuple(r["cluster_size_range"]), r["iteration"])
                 for r in results if all(k in r for k in ["scenario", "level", "cluster_size_range", "iteration"])}
    
    print(f"Resuming: {len(results)} results already done.")
    
    import threading
    results_lock = threading.Lock()
    
    all_combinations = []
    for scenario in SCENARIOS:
        if scenario not in clusters:
            continue
        for level in LEVEL_RANGE:
            for cluster_size_range in CLUSTER_SIZE_RANGE:
                all_combinations.append((
                    scenario, level, cluster_size_range,
                    aff_lookup, entity_lookup, emb_dict, clusters,
                    done_keys, results_lock, results
                ))

    print(
        f"Processing {len(all_combinations)} combinations "
        f"({len(LEVEL_RANGE)} levels × {len(CLUSTER_SIZE_RANGE)} cluster ranges × scenarios)"
    )

    print(f"Processing {len(all_combinations)} combinations with 256 workers")
    
    with ThreadPoolExecutor(max_workers=512) as executor:
        executor.map(process_combination, all_combinations)
    
    print(f"\nDone! Saved {len(results)} comparison sets.")


if __name__ == "__main__":
    main()
