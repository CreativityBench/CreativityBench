import os
from utils import batch_generate, load_json, save_json

PROMPT = """You are an expert at analyzing physical properties of object parts.

**Task:** Annotate physical attributes for the "{part_name}" of "{entity_name}".

**Context:**
- All parts: {all_parts}
- Connected to: {connected_to}
- Connection: {connection}

**Annotate these attributes:**
1. **Geometry & Shape:** shape, size, thickness (thin/medium/thick/NA), local_features
2. **Material & Structural:** material, rigidity (very rigid/rigid/semi-rigid/flexible/soft), durability (very fragile/fragile/normal/sturdy/very sturdy), elasticity (non-elastic/springy/stretchable/very stretchable), surface
3. **Mass:** weight (very light/light/moderate/heavy/very heavy)
4. **Others:** Other important attributes for affordance (1-2 sentences)
5. **Summary:** Comprehensive summary of all attributes

**Requirements:**
- Generate 2-3 physical attribute combinations for this part
- Each combination must be plausible, internally consistent, and diverse
- Include common variations (e.g., plastic vs steel)
- Include one unusual but plausible variation if it creates distinctive affordances
- All fields required; use "NA" only if truly not important for creative affordance
- Be assertive; avoid "might", "maybe", "could be"
- Ensure consistency (e.g., plastic → lighter, steel → heavier)

**Schema:**
```json
[
  {{
    "shape": "...",
    "size": "...",
    "thickness": "...",
    "local_features": "...",
    "material": "...",
    "rigidity": "...",
    "durability": "...",
    "elasticity": "...",
    "surface": "...",
    "weight": "...",
    "others": "...",
    "summary": "..."
  }}
]
```

**Example (glasses front_frame):**
```json
[
  {{
    "shape": "two connected loops and narrow bridge",
    "size": "hand-held",
    "thickness": "thin",
    "local_features": "two enclosed openings and flat front edge and narrow bridge span",
    "material": "plastic",
    "rigidity": "rigid",
    "durability": "normal",
    "elasticity": "non-elastic",
    "surface": "smooth",
    "weight": "very light",
    "others": "continuous rim can hook or hang on thin supports; bridge provides pinchable grip point",
    "summary": "A lightweight thin rigid plastic double-loop rim with enclosed openings and a pinchable bridge, smooth surfaced and easy to hang or hold."
  }},
  {{
    "shape": "two connected loops and narrow bridge",
    "size": "hand-held",
    "thickness": "thin",
    "local_features": "metal rim edge and small screw openings",
    "material": "steel",
    "rigidity": "very rigid",
    "durability": "sturdy",
    "elasticity": "non-elastic",
    "surface": "smooth and slightly cool",
    "weight": "light",
    "others": "conductive metal edge can transfer heat or cold; thin rim can fit into narrow slots",
    "summary": "A thin sturdy metal double-loop rim with small openings and a narrow bridge, rigid and thermally conductive with smooth edges."
  }}
]
```

First provide reasoning. Then output JSON array of 1-5 combinations."""

def main():
    input_file = "outputs/2_partonomy.json"
    output_file = "outputs/3_physical_attributes.json"
    
    entities = load_json(input_file)
    existing = load_json(output_file) if os.path.exists(output_file) else []
    completed_keys = {(e["entity_name"], e["part"]) for e in existing}
    
    tasks = [(e, p) for e in entities for p in e["parts"] if (e["entity_name"], p) not in completed_keys]
    print(f"Total parts: {sum(len(e['parts']) for e in entities)} | Completed: {len(existing)} | Pending: {len(tasks)}")
    
    if not tasks:
        print("All parts already processed")
        return
    
    prompts = [PROMPT.format(
        entity_name=e["entity_name"], part_name=p,
        all_parts=", ".join(e["parts"]),
        connected_to=", ".join(e["relations"][p]["connected_to"]),
        connection=e["relations"][p]["connection"]
    ) for e, p in tasks]
    
    results = existing.copy()
    
    def save_callback(data):
        new = [{"entity_name": tasks[i][0]["entity_name"], "scenario": tasks[i][0]["scenario"],
                "part": tasks[i][1], "all_parts": tasks[i][0]["parts"],
                "connected_to": tasks[i][0]["relations"][tasks[i][1]]["connected_to"],
                "connection": tasks[i][0]["relations"][tasks[i][1]]["connection"],
                "physical_attributes": r["data"]} 
               for i, r in enumerate(data) if r and "data" in r]
        save_json(results + new, output_file)
        print(f"\nSaved: {len(results) + len(new)} total")
    
    batch_results = batch_generate(prompts, max_workers=1024, save_callback=save_callback, save_interval=50, temperature=0.7)
    
    for i, r in enumerate(batch_results):
        if r and "data" in r:
            e, p = tasks[i]
            results.append({"entity_name": e["entity_name"], "scenario": e["scenario"],
                           "part": p, "all_parts": e["parts"],
                           "connected_to": e["relations"][p]["connected_to"],
                           "connection": e["relations"][p]["connection"],
                           "physical_attributes": r["data"]})
    
    save_json(results, output_file)
    success_rate = (len(results) - len(existing)) / len(tasks) * 100 if tasks else 100
    print(f"\nCompleted: {len(results)-len(existing)}/{len(tasks)} ({success_rate:.1f}%) | Total: {len(results)}")

if __name__ == "__main__":
    main()
