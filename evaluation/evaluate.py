import os
import json
import time
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from models import call_model
from utils import (
    build_static_prompt,
    build_cot_prompt,
    run_interactive,
    SYS_STATIC,
    SYS_COT,
    parse_response,
)

# ── MACROS (env vars with defaults) ───────────────────────────────────────────
TASK_FILE   = os.environ.get("TASK_FILE", os.path.join(os.path.dirname(__file__), "../dataset/sample_tasks.json"))
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "outputs"))
MODEL       = os.environ.get("MODEL", "gpt-5.2")
MODE        = os.environ.get("MODE", "static").lower()   # "static" | "cot" | "interactive"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "1024"))
SAVE_EVERY  = int(os.environ.get("SAVE_EVERY", "10"))
TEMPERATURES = [float(temp) for temp in os.environ.get("TEMPERATURES", "0.0,0.3,0.7").split(",")]

_SAVE_LOCK = threading.Lock()


# ── Per-sample runners ─────────────────────────────────────────────────────────
def _run_static(task: dict, temp: float) -> dict:
    prompt = build_static_prompt(task)
    text = call_model(
        MODEL,
        [{"role": "system", "content": SYS_STATIC},
         {"role": "user",   "content": prompt}],
        temperature=temp,
    )
    reasoning, parsed = parse_response(text)
    return {
        "temperature": temp,
        "reasoning":   reasoning,
        "gold_entity": parsed.get("gold_entity", ""),
        "gold_part":   parsed.get("gold_part",   ""),
        "how_to_use":  parsed.get("how_to_use",  ""),
    }


def _run_cot(task: dict, temp: float) -> dict:
    prompt = build_cot_prompt(task)
    text = call_model(
        MODEL,
        [{"role": "system", "content": SYS_COT},
         {"role": "user",   "content": prompt}],
        temperature=temp,
    )
    reasoning_text, parsed = parse_response(text)

    # Preferred schema: parsed["reasoning"] is a structured object.
    reasoning_obj = parsed.get("reasoning")
    if not isinstance(reasoning_obj, dict):
        # Backward-compatible fallback for older CoT schema.
        reasoning_obj = {}
        for k in [
            "task_goal",
            "success_condition",
            "identified_constraints",
            "candidate_parts",
            "reasoning_plan",
            "creative_reasoning_summary",
        ]:
            if k in parsed:
                reasoning_obj[k] = parsed[k]
        if not reasoning_obj and reasoning_text:
            reasoning_obj = {"raw_reasoning": reasoning_text}

    out = {
        "temperature": temp,
        "reasoning":   reasoning_obj,
        "gold_entity": parsed.get("gold_entity", ""),
        "gold_part":   parsed.get("gold_part",   ""),
        "how_to_use":  parsed.get("how_to_use",  ""),
    }
    return out


def _run_one(args):
    task, idx, temperature = args
    try:
        if MODE == "static":
            result = _run_static(task, temperature)
        elif MODE == "cot":
            result = _run_cot(task, temperature)
        elif MODE == "interactive":
            result = run_interactive(MODEL, task, temperature=temperature)
        else:
            raise ValueError(f"Unsupported MODE='{MODE}'. Expected one of: static, cot, interactive.")

        answer = result.get("answer", result)
        if not isinstance(answer, dict) or (not answer.get("gold_entity", "").strip() and not answer.get("gold_part", "").strip() and not answer.get("how_to_use", "").strip()):
            return task["task_id"], idx, None

        result["temperature"] = temperature
        result["sample_idx"] = idx
        return task["task_id"], idx, result
    
    except Exception as e:
        return task["task_id"], idx, None


def _is_result_filled(sample: dict) -> bool:
    if not isinstance(sample, dict):
        return False
    if "error" in sample:
        return False

    answer = sample.get("answer", sample)
    if not isinstance(answer, dict):
        return False
    if (
        not answer.get("gold_entity", "").strip()
        and not answer.get("gold_part", "").strip()
        and not answer.get("how_to_use", "").strip()
    ):
        return False
    return True


def _serialize_results(results: dict) -> dict:
    out = {}
    for task_id, samples in results.items():
        if not isinstance(samples, list):
            out[task_id] = samples
            continue
        filtered = [s for s in samples if _is_result_filled(s)]
        if filtered:
            out[task_id] = filtered
    return out


def _save_results(out_path: str, results: dict):
    """Atomically write current results to disk."""
    with _SAVE_LOCK:
        for attempt in range(3):
            try:
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                return
            except OSError as e:
                if e.errno != 24 or attempt == 2:
                    raise
                # If sockets/fds are temporarily saturated, back off briefly.
                time.sleep(0.5 * (attempt + 1))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Compute output path first
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    src       = os.path.splitext(os.path.basename(TASK_FILE))[0]
    model_tag = MODEL.replace("/", "_")
    out_path  = os.path.join(OUTPUT_DIR, f"{src}__{model_tag}__{MODE}.json")

    # Load existing results if file exists
    results: dict = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Loaded existing results from {out_path}")

    with open(TASK_FILE) as f:
        tasks = json.load(f)
    
    n_samples = len(TEMPERATURES)
    
    # Filter out already-completed task_id + sample_idx combinations.
    todo_list = []
    
    for task in tasks[:]:
        tid = task["task_id"]

        existing = results.get(tid)
        if not isinstance(existing, list):
            existing = []
        results[tid] = existing

        done_indices = set()
        remaining_temperatures = copy.deepcopy(TEMPERATURES)
        for entry in existing:
            if not isinstance(entry, dict):
                continue
            if "error" in entry:
                continue
            if not _is_result_filled(entry):
                continue
            temperature = entry.get("temperature")
            if temperature in remaining_temperatures:
                remaining_temperatures.remove(temperature)

        for idx, temperature in enumerate(remaining_temperatures):
            todo_list.append((task, idx+len(existing), temperature))

    if not todo_list:
        print("All tasks already completed. Nothing to do.")
        return

    print(f"Running {len(todo_list)} task/temperature combinations "
          f"({len(tasks) * n_samples - len(todo_list)} already done) "
          f"with {MAX_WORKERS} workers")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        completed = 0
        skipped = 0
        saved = 0
        for tid, idx, ans in tqdm(
            executor.map(_run_one, todo_list),
            total=len(todo_list),
            desc=f"[{MODE}] {MODEL}",
        ):
            if tid not in results:
                results[tid] = []

            if ans is None:
                skipped += 1
                completed += 1
                continue

            while len(results[tid]) <= idx:
                results[tid].append(None)
            results[tid][idx] = ans
            completed += 1
            saved += 1

            if SAVE_EVERY > 0 and saved % SAVE_EVERY == 0:
                _save_results(out_path, _serialize_results(results))

    _save_results(out_path, _serialize_results(results))
    
    if skipped:
        print(f"Skipped {skipped} due to null/error responses.")
    print(f"Saved → {out_path} ({completed - skipped} valid entries)")


if __name__ == "__main__":
    main()
