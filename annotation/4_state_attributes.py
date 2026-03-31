import os
from utils import batch_generate, load_json, save_json

PROMPT = """You are an expert at analyzing state and condition of object parts.

**Task:** Annotate state attributes for the "{part_name}" of "{entity_name}" with given physical attributes.

**Physical Attributes:**
{physical_attrs}

**Annotate these state attributes:**
1. **Access State:**
   - Visibility: visible, partially visible, hidden (relative to whole entity)
   - Availability: free, partially blocked (easily freed by hand), blocked (requires tools)
2. **Condition State:**
   - Moisture: dry, slightly wet, wet, NA
   - Temperature: cold, slightly cold, slightly hot, hot, NA (room temp)
3. **Internal State:**
   - Internal: empty, partially filled, full, NA (physical or abstract capacity)
4. **Others:** Other important state attributes (1-2 sentences)
5. **Summary:** Comprehensive summary of all state attributes

**Requirements:**
- Generate 2-3 state attribute combinations for this part
- Must be consistent with physical attributes
- Include common/typical state and unusual but plausible states
- All combinations must be plausible, consistent, and diverse
- All fields required; use "NA" if not important
- Be assertive; avoid "might", "maybe"
- States should not contradict physical attributes

**Schema:**
```json
[
  {{
    "visibility": "...",
    "availability": "...",
    "moisture": "...",
    "temperature": "...",
    "internal": "...",
    "others": "...",
    "summary": "..."
  }}
]
```

**Example (vacuum bag - flexible, dry):**
```json
[
  {{
    "visibility": "hidden",
    "availability": "partially blocked",
    "moisture": "dry",
    "temperature": "NA",
    "internal": "partially filled",
    "others": "Seated inside a closed compartment; compartment cover is latched shut; bag collar aligned on plastic inlet mount; bag material flexible but holds shape from airflow.",
    "summary": "Hidden in latched compartment and partially blocked by cover and inlet mount; dry at room temperature; partially filled; flexible bag seated on collar mount."
  }},
  {{
    "visibility": "hidden",
    "availability": "blocked",
    "moisture": "slightly wet",
    "temperature": "NA",
    "internal": "partially filled",
    "others": "Contents clumped and tacky, bag adheres to compartment liner; bag surface damp and slightly softened; collar stuck on inlet mount and won't release with simple pull.",
    "summary": "Hidden and blocked by adhesion and stuck collar mount; slightly wet at room temperature; partially filled with clumped debris preventing easy hand removal."
  }}
]
```

First provide reasoning considering physical attributes. Then output JSON array of 1-4 combinations."""

def format_physical_attrs(attrs):
    return "\n".join([f"  - {k}: {v}" for k, v in attrs.items() if k != "summary"])

def main():
    input_file = "outputs/4_physical_combined.json"
    output_file = "outputs/5_state_attributes.json"
    
    entities = load_json(input_file)
    existing = load_json(output_file) if os.path.exists(output_file) else []
    completed_keys = {(e["entity_name"], e["part_name"], str(e["physical_attributes"])) for e in existing}
    
    tasks = [(e, p) for e in entities for p in e["parts"] 
             if (e["entity_name"], p["part_name"], str(p["physical_attributes"])) not in completed_keys]
    total_parts = sum(len(e["parts"]) for e in entities)
    print(f"Total parts: {total_parts} | Completed: {len(existing)} | Pending: {len(tasks)}")
    
    if not tasks:
        print("All parts already processed")
        return
    
    prompts = [PROMPT.format(entity_name=e["entity_name"], part_name=p["part_name"],
               physical_attrs=format_physical_attrs(p["physical_attributes"])) for e, p in tasks]
    results = existing.copy()
    
    def save_callback(data):
        new = [{"entity_name": tasks[i][0]["entity_name"], "scenario": tasks[i][0]["scenario"],
                "part_name": tasks[i][1]["part_name"], "connected_to": tasks[i][1]["connected_to"],
                "connection": tasks[i][1]["connection"], "physical_attributes": tasks[i][1]["physical_attributes"],
                "state_attributes": r["data"]} for i, r in enumerate(data) if r and "data" in r]
        save_json(results + new, output_file)
        print(f"\nSaved: {len(results) + len(new)} total")
    
    batch_results = batch_generate(prompts, max_workers=1024, save_callback=save_callback, save_interval=50, temperature=0.7)
    
    for i, r in enumerate(batch_results):
        if r and "data" in r:
            e, p = tasks[i]
            results.append({"entity_name": e["entity_name"], "scenario": e["scenario"],
                           "part_name": p["part_name"], "connected_to": p["connected_to"],
                           "connection": p["connection"], "physical_attributes": p["physical_attributes"],
                           "state_attributes": r["data"]})
    
    save_json(results, output_file)
    success_rate = (len(results) - len(existing)) / len(tasks) * 100 if tasks else 100
    print(f"\nCompleted: {len(results)-len(existing)}/{len(tasks)} ({success_rate:.1f}%) | Total: {len(results)}")

if __name__ == "__main__":
    main()
