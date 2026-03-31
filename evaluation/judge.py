import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from models import call_model
from utils import parse_json
from tqdm import tqdm

# ── MACROS ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "outputs"))
JUDGED_OUTPUT_DIR = os.environ.get("JUDGED_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "judged_outputs"))
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", os.environ.get("MODEL", "gpt-5.2"))
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
MAX_WORKERS = int(os.environ.get("JUDGE_MAX_WORKERS", "20"))
FILE_MAX_WORKERS = int(os.environ.get("JUDGE_FILE_MAX_WORKERS", "1"))

TASK_DIR_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "../dataset"),
    os.path.join(os.path.dirname(__file__), "../task_creation/outputs"),
]
ENTITY_LOOKUP_FILE = os.path.join(os.path.dirname(__file__), "../task_creation/outputs/1_entity_lookup.json")

JUDGE_FIELDS = [
    "environment_condition_covered_reason",
    "environment_condition_covered",
    "use_condition_covered_reason",
    "use_condition_covered",
    "recipient_condition_covered_reason",
    "recipient_condition_covered",
    "attributes_grounding_reason",
    "attributes_grounding",
    "prediction_correctness_reason",
    "prediction_correctness",
    "action_feasibility_reason",
    "action_feasibility"
]

ENTITY_LOOKUP_CACHE = None


def try_int(val, default=10):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _is_na(v):
    return str(v).strip().upper() == "NA" or str(v).strip() == ""


def _extract_answer(sample):
    if isinstance(sample, dict) and "answer" in sample and isinstance(sample["answer"], dict):
        ans = sample["answer"]
        return ans.get("gold_entity", ""), ans.get("gold_part", ""), ans.get("how_to_use", "")
    if isinstance(sample, dict):
        return sample.get("gold_entity", ""), sample.get("gold_part", ""), sample.get("how_to_use", "")
    return "", "", ""


def _name_core(text):
    s = str(text or "").lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _word_f1(a, b):
    ta = _name_core(a).split()
    tb = _name_core(b).split()
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    inter = sum((ca & cb).values())
    if inter == 0:
        return 0.0
    p = inter / len(ta)
    r = inter / len(tb)
    return (2 * p * r) / (p + r)


def _best_by_f1(pred, candidates):
    best = ""
    best_f1 = -1.0
    for c in candidates:
        s = _word_f1(pred, c)
        if s > best_f1:
            best_f1 = s
            best = c
    return best


def _is_sample_judged(sample):
    if not isinstance(sample, dict):
        return False
    required = ["task_uid", "entity_correct", "part_correct", "prediction_relation", "llm_judge"]
    return all(k in sample for k in required)


def _is_zero_temperature_sample(sample):
    if not isinstance(sample, dict):
        return False
    try:
        return float(sample.get("temperature")) == 0.0
    except (TypeError, ValueError):
        return False


def _resolve_task_file(src_name):
    fn = f"{src_name}.json"
    for d in TASK_DIR_CANDIDATES:
        p = os.path.abspath(os.path.join(d, fn))
        if os.path.exists(p):
            return p
    return None


def _load_entity_lookup():
    global ENTITY_LOOKUP_CACHE
    if ENTITY_LOOKUP_CACHE is None:
        with open(ENTITY_LOOKUP_FILE) as f:
            raw = json.load(f)
        ENTITY_LOOKUP_CACHE = list(raw.values()) if isinstance(raw, dict) else raw
    return ENTITY_LOOKUP_CACHE


def _get_part_attrs(task, pred_entity, pred_part):
    scenario = str(task.get("scenario", "") or "")
    entity_core = _name_core(pred_entity)
    part_core = _name_core(pred_part)

    for item in _load_entity_lookup():
        if not isinstance(item, dict):
            continue
        if str(item.get("scenario", "") or "") != scenario:
            continue
        if _name_core(item.get("entity_name", "")) != entity_core:
            continue
        for part in item.get("parts", []):
            if isinstance(part, dict) and _name_core(part.get("part_name", "")) == part_core:
                return (
                    part.get("physical_attributes", {}),
                    part.get("state_attributes", {}),
                )
    return {}, {}


def _call_judge_llm(prompt):
    text = call_model(
        JUDGE_MODEL,
        [{"role": "user", "content": prompt}],
        temperature=JUDGE_TEMPERATURE,
    )
    parsed = parse_json(text)
    return parsed if isinstance(parsed, dict) else {}


def _normalize_llm_judge(obj, force_na_flags={}):
    out = {k: obj.get(k, "") for k in JUDGE_FIELDS}
    for key, flag in force_na_flags.items():
        if flag:
            out[f"{key}_reason"] = "NA by gold affordance condition."
            out[key] = "NA"

    for k in ["environment_condition_covered", "use_condition_covered", "recipient_condition_covered"]:
        if str(out[k]) not in ["0", "1", "2", "NA"]:
            out[k] = "NA" if force_na_flags.get(k, 0) else 0
        if str(out[k]) in ["0", "1", "2"]:
            out[k] = int(out[k])
    for k in ["attributes_grounding", "prediction_correctness", "action_feasibility"]:
        if str(out[k]) not in ["0", "1", "2"]:
            out[k] = 0
        if str(out[k]) in ["0", "1", "2"]:
            out[k] = int(out[k])
    return out


def _prediction_relation(task, pred_entity, pred_part, entity_correct, part_correct):
    if entity_correct and part_correct:
        return "gold", None

    entity_obj = next((e for e in task.get("entities", []) if e.get("name") == pred_entity), None)
    if not entity_obj:
        return "not similar", None
    judge_output = entity_obj.get("judge_output", {}).get(pred_part)
    if not isinstance(judge_output, dict):
        return "not similar", None

    similar = judge_output.get("similar_affordance") == "Yes"
    not_change = judge_output.get("gold_change") == "No"
    if similar and not_change:
        d = try_int(judge_output.get("decision_making_difficulty"), default=10)
        return f"similar-{d}" if d <= 5 else "similar", judge_output
    return "not similar", judge_output


def _prompt_correct(task, how_to_use):
    gold = task["golds"][0]
    aff = gold.get("gold_affordance", {})
    force_na_flags = {
        "environment_condition_covered": _is_na(aff.get("environment_condition", "")),
        "use_condition_covered": _is_na(aff.get("use_condition", "")),
        "recipient_condition_covered": _is_na(aff.get("recipient_condition", "")),
    }

    prompt = f"""You are a strict evaluator for embodied task instructions.
Evaluate whether the predicted "how_to_use" is feasible compared to the gold affordance and gold solution.
Use evidence-based judgments for each field. Please follow the rubric strictly.

Task:
{task.get("task", "")}

Gold affordance JSON:
{json.dumps(gold.get("gold_affordance", {}), ensure_ascii=False, indent=2)}

Gold solution JSON:
{json.dumps(task.get("solution", {}), ensure_ascii=False, indent=2)}

Predicted how_to_use:
{how_to_use}

Field-by-field rubric when judging the predicted "how_to_use":
1) environment_condition_covered (0/1/2/"NA"):
- 0: required external environment setup in gold is not mentioned at all in the predicted "how_to_use".
- 1: required external environment setup in gold is mentioned, but not all are covered in the predicted "how_to_use".
- 2: required external environment setup in gold is covered well and reasonable in the predicted "how_to_use".
- NA: only if in the gold affordance annotation, the environment_condition is NA.

2) use_condition_covered (0/1/2/"NA"):
- 0: required preparation/access of the tool-part is not mentioned at all in the predicted "how_to_use".
- 1: required preparation/access of the tool-part is mentioned, but not all are covered in the predicted "how_to_use".
- 2: required preparation/access of the tool-part is covered well and reasonable in the predicted "how_to_use".
- NA: only if in the gold affordance annotation, the use_condition is NA.

3) recipient_condition_covered (0/1/2/"NA"):
- 0: recipient-side prerequisites are not mentioned at all in the predicted "how_to_use".
- 1: recipient-side prerequisites are mentioned, but not all are covered in the predicted "how_to_use".
- 2: recipient-side prerequisites are covered well and reasonable.
- False: recipient assumptions are not met in the predicted "how_to_use".
- NA: only if in the gold affordance annotation, the recipient_condition is NA.

4) attributes_grounding (0/1/2): [compare to the key enabling attributes]
- 0: the predicted action is not grounded in the key enabling attributes of the gold affordance, or it violates some attributes of the part.
- 1: the predicted action is mostly grounded in the key enabling attributes of the gold affordance, but not all are covered or implied.
- 2: the predicted action is fully grounded in the key enabling attributes of the gold affordance, and most of them are explicitly mentioned, covered or implied.

5) prediction_correctness (0/1/2): [compare to the gold solution]
- 0: compared with the gold solution the predicted "how_to_use" is completely wrong, not working at all.
- 1: compared with the gold solution the predicted "how_to_use" is partially workable but missing crucial details/order/precision.
- 2: compared with the gold solution the predicted "how_to_use" is operationally correct and complete, almost completely aligned with gold solution.

6) action_feasibility (0/1/2): [not compare to the gold solution, but focus on the action itself]
- 0: the action itself is physically impossible, not working at all, very unlikely to be used in practice, or unsafe.
- 1: the action itself is partially workable but still there are some steps that are not plausible, aligned with common sense or not feasible.
- 2: the action itself is operationally correct and complete, almost completely aligned with common sense and feasible in practice.

Evidence policy:
- Every *_reason must quote concrete evidence from predicted "how_to_use" and other given relevant information.
- If evidence is unclear/missing, default to a stricter outcome (lower score).
- Make your reasoning clear, concise, and to the point for each field, and your score should be based on the evidence.

Please make sure to only return a valid JSON with exactly these 12 fields (no extras, no markdown).
{{
    "environment_condition_covered_reason": "...",
    "environment_condition_covered": 0/1/2/"NA",
    "use_condition_covered_reason": "...",
    "use_condition_covered": 0/1/2/"NA",
    "recipient_condition_covered_reason": "...",
    "recipient_condition_covered": 0/1/2/"NA",
    "attributes_grounding_reason": "...",
    "attributes_grounding": 0/1/2,
    "prediction_correctness_reason": "...",
    "prediction_correctness": 0/1/2,
    "action_feasibility_reason": "...",
    "action_feasibility": 0/1/2,
}}
"""
    return prompt, force_na_flags


def _prompt_incorrect(task, pred_entity, pred_part, how_to_use):
    pred_physical_attrs, pred_state_attrs = _get_part_attrs(task, pred_entity, pred_part)
    prompt = f"""You are a strict evaluator for practical goal completion.
The predicted entity/part is NOT exactly gold.
Judge whether the predicted "how_to_use" can still accomplish the task reasonably and safely.

Task:
{task.get("task", "")}

Gold solution JSON (only for reference):
{json.dumps(task.get("solution", {}), ensure_ascii=False, indent=2)}

Physical attributes of the predicted part:
{json.dumps(pred_physical_attrs, ensure_ascii=False, indent=2)}

State attributes of the predicted part:
{json.dumps(pred_state_attrs, ensure_ascii=False, indent=2)}

Predicted entity to use: {pred_entity}
Predicted part to use: {pred_part}
Predicted how to use the part:
{how_to_use}

Field-by-field rubric when judging the predicted "how_to_use":
1) environment_condition_covered (0/1/2/"NA"): [Do NOT directly compare with the gold, it's only for reference]
- "NA": in your reasoning, deeply consider for the predicted use of the part to accomplish the task, if the external environment setup is needed; if not, then please give "NA".
- 0: the environment setup is needed but in the predicted use, it is not mentioned or covered at all.
- 1: the environment setup is needed but in the predicted use, it is mentioned but not all are covered.
- 2: the environment setup is needed but in the predicted use, it is covered well and reasonable, aligning with common sense.

2) use_condition_covered (0/1/2/"NA"): [Do NOT directly compare with the gold, it's only for reference]
- "NA": in your reasoning,deeply consider for the predicted use of the part to accomplish the task, if the preparation/access of the part is needed; if not, then please give "NA".
- 0: the preparation/access of the part is needed but in the predicted use, it is not mentioned or covered at all.
- 1: the preparation/access of the part is needed but in the predicted use, it is mentioned but not all are covered.
- 2: the preparation/access of the part is needed but in the predicted use, it is covered well and reasonable, aligning with common sense.

3) recipient_condition_covered (0/1/2/"NA"): [Do NOT directly compare with the gold, it's only for reference]
- "NA": in your reasoning, deeply consider for the predicted use of the part to accomplish the task, if the recipient-side prerequisites are needed; if not, then please give "NA".
- 0: the recipient-side prerequisites are needed but in the predicted use, it is not mentioned or covered at all.
- 1: the recipient-side prerequisites are needed but in the predicted use, it is mentioned but not all are covered.
- 2: the recipient-side prerequisites are needed but in the predicted use, it is covered well and reasonable, aligning with common sense.

4) attributes_grounding (0/1/2): [Compare with the physical and state attributes of the predicted part]
- 0: the predicted action is not grounded in the key enabling attributes of the part, or it violates some attributes of the part.
- 1: the predicted action is mostly grounded in the key enabling attributes of the gold affordance, but not all are covered or implied.
- 2: the predicted action is fully grounded in the key enabling attributes of the gold affordance, and most of them are explicitly mentioned, covered or implied.

5) action_feasibility (0/1/2): [Please only focus on the predicted action itself]
- 0: the action itself is physically impossible, not working at all, very unlikely to be used in practice, or unsafe.
- 1: the action itself is partially workable but still there are some steps that are not plausible, aligned with common sense or not feasible.
- 2: the action itself is operationally correct and complete, almost completely aligned with common sense and feasible in practice.

Evidence policy:
- Every *_reason must quote concrete evidence from predicted how to use the part and other given relevant information.
- If evidence is unclear/missing, default to a stricter outcome (lower score).
- Make your reasoning clear, concise, and to the point for each field, and your score should be based on the evidence.

Please make sure to only return a valid JSON with exactly these 12 fields (no extras, no markdown).
{{
    "environment_condition_covered_reason": "...",
    "environment_condition_covered": 0/1/2/"NA",
    "use_condition_covered_reason": "...",
    "use_condition_covered": 0/1/2/"NA",
    "recipient_condition_covered_reason": "...",
    "recipient_condition_covered": 0/1/2/"NA",
    "attributes_grounding_reason": "...",
    "attributes_grounding": 0/1/2,
    "action_feasibility_reason": "...",
    "action_feasibility": 0/1/2,
}}
"""
    return prompt, {}


def _prompt_prediction_reasonability(task, pred_entity, pred_part, how_to_use, pred_comp):
    prompt = f"""You are a strict evaluator comparing predicted substitution vs ground-truth preference.
You are given the model prediction and the a ground-truth comparison judgment for that predicted entity+part.
Judge how convincing the model prediction remains AFTER reading the comparison evidence.

Task:
{task.get("task", "")}

Predicted entity to use: {pred_entity}
Predicted part to use: {pred_part}
Predicted how to use the part:
{how_to_use}

Ground-truth comparison judgment for the predicted entity+part:
{json.dumps(pred_comp if pred_comp else {}, ensure_ascii=False, indent=2)}

Return ONLY JSON:
{{
  "prediction_reasonable_reason": "...",
  "prediction_reasonable_level": 0 or 1 or 2
}}

Scoring rubric:
- 0: prediction is somewhat reasonable, but ground truth is clearly better after reading the comparison judgment.
- 1: prediction has meaningful merit; trade-offs are close and prediction may be acceptable after reading the comparison judgment.
- 2: prediction is strongly convincing and should replace current gold choice after reading the comparison judgment.

Reasoning requirements:
- Reference available comparison aspects (accessibility, side-effects, willingness, commonness, safety, etc.).
- Consider whether predicted "how to use the part" strengthens or weakens the substitution.
- Penalize unsupported claims.

Please make sure to only return a valid JSON with exactly these 2 fields (no extras, no markdown)."""
    return prompt


def _judge_one(task, sample):
    pred_entity, pred_part, how_to_use = _extract_answer(sample)
    gold = task["golds"][0]
    gold_entity = gold.get("gold_entity", "")
    gold_part = gold.get("gold_part", "")
    
    if True:
        exact_entity = pred_entity == gold_entity
        exact_part = pred_part == gold_part

        entity_candidates = [str(e.get("name", "")) for e in task.get("entities", []) if isinstance(e, dict) and str(e.get("name", ""))]
        if gold_entity and gold_entity not in entity_candidates:
            entity_candidates.append(gold_entity)

        best_entity = _best_by_f1(pred_entity, entity_candidates)
        f1_entity_ok = _name_core(best_entity) == _name_core(gold_entity)

        f1_part_ok = False
        best_entity_obj = next((e for e in task.get("entities", []) if isinstance(e, dict) and str(e.get("name", "")) == best_entity), None)
        part_candidates = []
        if isinstance(best_entity_obj, dict):
            judge_output = best_entity_obj.get("judge_output", {})
            if isinstance(judge_output, dict):
                part_candidates = [str(k) for k in judge_output.keys() if str(k)]
        if not part_candidates and gold_part:
            part_candidates = [str(gold_part)]
        best_part = _best_by_f1(pred_part, part_candidates)
        f1_part_ok = _name_core(best_part) == _name_core(gold_part)

        # ground the prediction to the best entity and part
        entity_correct = exact_entity or f1_entity_ok
        part_correct = exact_part or f1_part_ok
        pred_entity = best_entity
        pred_part = best_part

    relation, pred_comp = _prediction_relation(task, pred_entity, pred_part, entity_correct, part_correct)
    if relation != "gold" and pred_comp is None:
        relation = "hallucination"
        pred_comp = "The predicted entity or part does not really exist, so there is no existing comparison judgment. This may be due to the model hallucinating or the it does not output the entity or the part's name faithfully."

    llm_detail = {
        "environment_condition_covered_reason": "NA",
        "environment_condition_covered": 0,
        "use_condition_covered_reason": "NA",
        "use_condition_covered": 0,
        "recipient_condition_covered_reason": "NA",
        "recipient_condition_covered": 0,
        "attributes_covered_reason": "NA",
        "attributes_grounding": 0,
        "prediction_correctness_reason": "NA",
        "prediction_correctness": 0,
        "action_feasibility_reason": "NA",
        "action_feasibility": 0,
    }
    
    if how_to_use:
        if entity_correct and part_correct:
            p, force_na = _prompt_correct(task, how_to_use)
            llm_detail = _normalize_llm_judge(_call_judge_llm(p), force_na_flags=force_na)
            prediction_reasonability = {}
        
        if not entity_correct or not part_correct:
            p, force_na = _prompt_incorrect(task, pred_entity, pred_part, how_to_use)
            llm_detail = _normalize_llm_judge(_call_judge_llm(p))

            p2 = _prompt_prediction_reasonability(task, pred_entity, pred_part, how_to_use, pred_comp)
            r2 = _call_judge_llm(p2)
            prediction_reasonability = {
                "prediction_reasonable_reason": r2.get("prediction_reasonable_reason", ""),
                "prediction_reasonable_level": r2.get("prediction_reasonable_level", 0),
            }

    judged = dict(sample) if isinstance(sample, dict) else {"raw_sample": sample}
    judged.update({
        "task_uid": task.get("task_id", ""),
        "setting": task.get("setting", None),
        "best_matching_entity": best_entity,
        "best_matching_part": best_part,
        "entity_correct": entity_correct,
        "part_correct": part_correct,
        "prediction_relation": relation,
        "predicted_entity_part_judge_output": pred_comp,
        "llm_judge": llm_detail,
        "prediction_reasonability_judge": prediction_reasonability,
    })
    return judged


def _judge_output_file(in_path):
    stats = {
        "file": os.path.basename(in_path),
        "total_samples": 0,
        "already_judged": 0,
        "newly_judged": 0,
        "not_judged": [],  # [{task_id, sample_idx, reason}]
        "skipped_fully_judged": False,
    }

    base = os.path.basename(in_path)
    src = base.split("__", 1)[0]
    task_file = _resolve_task_file(src)
    if task_file is None:
        print(f"Skip {base}: cannot find task source file {src}.json")
        stats["not_judged"].append({"task_id": "*", "sample_idx": -1, "reason": "task_source_file_not_found"})
        return stats

    with open(task_file) as f:
        tasks = json.load(f)
    task_map = {t["task_id"]: t for t in tasks}

    with open(in_path) as f:
        outputs = json.load(f)

    os.makedirs(JUDGED_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(JUDGED_OUTPUT_DIR, base)
    existing_judged = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing_judged = json.load(f)

    jobs = []
    judged = {}
    
    all_tasks = 0
    zero_temperature_tasks = 0
    
    for task_id, samples in outputs.items():
        zero_idx = next((idx for idx, sample in enumerate(samples) if _is_zero_temperature_sample(sample)), None)
        
        all_tasks += 1
        if zero_idx is not None:
            zero_temperature_tasks += 1
        elif len(samples) > 0:
            zero_idx = 0
        
        existing = existing_judged.get(task_id, [])
        existing_zero = None
        for e in existing:
            if isinstance(e, dict) and e.get("llm_judge", {}) != {}:
                existing_zero = e
                break
        judged[task_id] = [existing_zero] if (zero_idx is not None or existing_zero is not None) else []

        task = task_map.get(task_id)
        if not task:
            print(f"Task {task_id} not found in task file {task_file}!")
            continue
        
        stats["total_samples"] += 1
        if _is_sample_judged(existing_zero):
            stats["already_judged"] += 1
        else:
            jobs.append((task_id, zero_idx, task, samples[zero_idx]))
    
    print(f"In file {base}, All tasks: {all_tasks}, zero temperature tasks: {zero_temperature_tasks}, already judged: {stats['already_judged']}")

    if not jobs and (stats["already_judged"] == stats["total_samples"]):
        stats["skipped_fully_judged"] = True
        print(f"Skip {base}: fully judged already.")
        return stats
    
    if jobs:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {
                ex.submit(_judge_one, task, sample): (task_id, idx)
                for task_id, idx, task, sample in jobs
            }
            for fut in tqdm(
                as_completed(futs),
                total=len(futs),
                desc=f"[inner] {base}",
                leave=False,
            ):
                task_id, idx = futs[fut]
                try:
                    judged[task_id] = [fut.result()]
                    stats["newly_judged"] += 1
                except Exception as e:
                    stats["not_judged"].append({
                        "task_id": task_id, "sample_idx": idx, "reason": f"judge_exception: {e}"
                    })
                    if not judged[task_id] or judged[task_id][0] is None:
                        src_sample = outputs.get(task_id, [])[idx]
                        judged[task_id] = [dict(src_sample) if isinstance(src_sample, dict) else {"raw_sample": src_sample}]

    for task_id, samples in outputs.items():
        if judged[task_id]:
            continue
        zero_idx = next((idx for idx, sample in enumerate(samples) if _is_zero_temperature_sample(sample)), None)
        if zero_idx is None:
            continue
        s = samples[zero_idx]
        judged[task_id] = [dict(s) if isinstance(s, dict) else {"raw_sample": s}]
        stats["not_judged"].append({
            "task_id": task_id, "sample_idx": zero_idx, "reason": "not_judged_unknown"
        })

    with open(out_path, "w") as f:
        json.dump(judged, f, ensure_ascii=False, indent=2)
    print(
        f"Saved judged file: {out_path} "
        f"(already={stats['already_judged']}, newly={stats['newly_judged']}, "
        f"not_judged={len(stats['not_judged'])})"
    )
    return stats


def main():
    os.makedirs(JUDGED_OUTPUT_DIR, exist_ok=True)
    files = sorted(
        os.path.join(OUTPUT_DIR, fn)
        for fn in os.listdir(OUTPUT_DIR)
        if fn.endswith(".json")
    )

    print(
        f"Judging {len(files)} files with model={JUDGE_MODEL} "
        f"(outer_workers={FILE_MAX_WORKERS}, inner_workers={MAX_WORKERS})"
    )
    
    all_stats = []
    with ThreadPoolExecutor(max_workers=FILE_MAX_WORKERS) as ex:
        futs = [ex.submit(_judge_output_file, fp) for fp in files]
        for fut in as_completed(futs):
            all_stats.append(fut.result())

    reason_count = {}
    total_not_judged = 0
    skipped_files = 0
    for st in all_stats:
        if not st:
            continue
        if st.get("skipped_fully_judged"):
            skipped_files += 1
        for item in st.get("not_judged", []):
            total_not_judged += 1
            reason = item["reason"]
            reason_count[reason] = reason_count.get(reason, 0) + 1

    if reason_count:
        print("Not judged by reason:")
        for r, c in sorted(reason_count.items(), key=lambda x: (-x[1], x[0])):
            print(f"- {r}: {c}")
        print("\nNot judged sample list:")
        for st in all_stats:
            if not st:
                continue
            for item in st.get("not_judged", []):
                print(
                    f"- file={st['file']} task_id={item['task_id']} "
                    f"sample_idx={item['sample_idx']} reason={item['reason']}"
                )
    print("\n=== Final Statistics ===")
    print(f"Files total: {len(files)}")
    print(f"Files skipped (fully judged): {skipped_files}")
    print(f"Not judged samples total: {total_not_judged}")


if __name__ == "__main__":
    main()
