import os
from utils import batch_generate, load_json, save_json

PROMPT = """You are an expert at analyzing object structure and component parts.

**Task:** Annotate the partonomy graph for "{entity_name}" (found in {scenario}).

**Requirements:**
1. List ALL the common parts (both externally visible and internally hidden ones)
2. Limit to 8 parts maximum, can be less if the object is simple and parts are obvious
3. Your parts should better have explicit boundaries which divide the object's different function units or affordance mechanisms
4. Include components that may have useful affordances themselves (e.g. screws, batteries, etc.).
5. Parts must be non-overlapping and cover the whole object (e.g. a blade contains cutting edge, so it's one part).
6. Imagine a specific entity when you annotate the parts, don't include any optional parts or uncertain wording, all parts should be necessary and essential to the entity's function.
7. Parts with same function differing only by position or direction should be annotated as ONE part
8. For each part: specify connected parts and describe function/connection

**Schema:**
```json
{{
  "entity_name": "...",
  "parts": ["part1", "part2", ...],
  "relations": {{
    "part1": {{
      "connected_to": ["part2", "part3"],
      "connection": "Description of part1, its function, and connection"
    }},
    ...
  }}
}}
```

**Example (prescription reading glasses):**
```json
{{
  "entity_name": "prescription reading glasses",
  "parts": ["front_frame", "lenses", "hinge_mechanisms", "temple_arms", "temple_tips"],
  "relations": {{
    "front_frame": {{
      "connected_to": ["lenses", "hinge_mechanisms"],
      "connection": "Main rigid front structure; provides lens openings/retention geometry; provides hinge mounting points."
    }},
    "lenses": {{
      "connected_to": ["front_frame"],
      "connection": "Two optical elements seated/retained by the front frame."
    }},
    ...
  }}
}}
```

First imagine a specific entity when you annotate the parts, and then provide reasoning about its structure and key parts, and then output the parts and relations JSON."""

def main():
    input_file = "1_sample_entities.json"
    output_file = "outputs/2_partonomy.json"
    
    entities = load_json(input_file)
    existing = load_json(output_file) if os.path.exists(output_file) else []
    completed_names = {e["entity_name"] for e in existing}
    
    pending = [e for e in entities if e["name"] not in completed_names]
    print(f"Total: {len(entities)} | Completed: {len(existing)} | Pending: {len(pending)}")
    
    if not pending:
        print("All entities already processed")
        return
    
    prompts = [PROMPT.format(entity_name=e["name"], scenario=e["scenario"]) for e in pending]
    results = existing.copy()
    
    # Create mapping from entity name to scenario
    entity_scenario_map = {e["name"]: e["scenario"] for e in pending}
    
    def save_callback(data):
        new_results = []
        for r in data:
            if r and "data" in r:
                e = r["data"]
                scenario = entity_scenario_map.get(e["entity_name"], "unknown")
                new_results.append({
                    "entity_name": e["entity_name"],
                    "scenario": scenario,
                    "parts": e["parts"],
                    "relations": e["relations"]
                })
        current = results + new_results
        save_json(current, output_file)
        print(f"\nSaved: {len(current)} total")
    
    batch_results = batch_generate(prompts, max_workers=32, save_callback=save_callback, save_interval=10)
    
    for r in batch_results:
        if r and "data" in r:
            e = r["data"]
            scenario = entity_scenario_map.get(e["entity_name"], "unknown")
            results.append({
                "entity_name": e["entity_name"],
                "scenario": scenario,
                "parts": e["parts"],
                "relations": e["relations"]
            })
    
    save_json(results, output_file)
    success_rate = len(new_results) / len(pending) * 100 if pending else 100
    print(f"\nCompleted: {len(new_results)}/{len(pending)} ({success_rate:.1f}%) | Total: {len(results)}")

if __name__ == "__main__":
    main()
