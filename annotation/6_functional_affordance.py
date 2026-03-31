import os
from utils import batch_generate, load_json, save_json, save_json_append

PROMPT = """You are an expert at identifying functional affordances based on attributes.

**Task:** Annotate functional affordances for the "{part_name}". This part belongs to the entity "{entity_name}" in the scenario "{scenario}".

**Physical Attributes of Part "{part_name}":**
{physical_attrs}

**State Attributes of Part "{part_name}":**
{state_attrs}

**Instructions:**
Identify 6 different functional affordances for this part. For each:

1. **Use Condition:** What preparation is needed to access this affordance?
   - "NA" if part is free and directly usable
   - Otherwise describe preparation steps (e.g., "break the lens", "remove from case")
   - Based on visibility/availability from state attributes

2. **Environment Condition:** What environmental conditions are needed for this affordance?
   - Focus on scenario/environment requirements (e.g., "lighting available", "power source nearby")
   - NOT about the part itself or recipient, but about external conditions
   - "NA" if no special environment needed

3. **Attribute:** What attributes enable this affordance?
   - List relevant attributes that are needed to be considered for this affordance; your stated attributes must be derived from the given physical + state attributes above; do not introduce new attributes that are not provided
   - Format: JSON list of lists `[["attribute statement", "physical/state", "visual/text", "explanation why visual/text"], ...]`
   - "physical" = attribute statement derived from physical attributes given above; "state" = attribute statement derived from state attributes given above
   - "visual" = this attribute can be clearly illustrated in an image without ambiguity; "text" = this attribute needs text description to clarify (e.g., temperature, texture, hidden features)
   - The explanation should be your reason why this attribute needs to be illustrated using text to clarify or only visual signal is enough to convey the information.
   - Example: `[["the material of the lens is glass", "physical", "visual", "Glass material is visually identifiable by its transparency and reflective surface"], ["the surface is smooth", "physical", "text", "Smoothness is a tactile property difficult to convey visually alone"]]`

4. **Affordance:** What can this part be used for?
   - Brief, clear description of the function/purpose
   - Can be original normal function (must have at least one original normal function) OR creative alternative use
   - Must be plausible and grounded in this part's given physical and state attributes above
   - Must act upon a recipient to take effect: passive or decorative roles (e.g., "use as decoration", "display for aesthetics") are NOT valid affordances since they do not act upon anything
   - Write at a high, general level: describe the functional capability, not a specific scenario (good: "dig into soft material"; bad: "use as a shovel in a garden").
   - If the use is too rare, niche, or uncommon to be practically meaningful, do not include it.
   - If the affordance requires specific conditions to work, those must be reflected faithfully in use_condition, environment_condition, and recipient_condition

5. **Level:** Categorize the affordance type and suitability
   - "Normal 0" = Original intended normal function of the part within the entity (must have at least one original normal function)
   - "Emergency 1-5" = Creative/alternative use
     - 1 = Rarely used, only if nothing else available; difficult to access or imperfect; in real life people rarely use this affordance this way
     - 2 = In between 1 and 3
     - 3 = Moderately suitable, could work in urgent situations; in real life people may use this affordance a few times in real emergencies but not too much
     - 4 = In between 3 and 5
     - 5 = Highly suitable, easy and natural to use; very effective; in real life people are willing to use this affordance this way most of the time
   - Take into consideration if normally people can have access to this affordance, whether this part needs to be detached from the whole entity to use this affordance, whether to use this affordance will irreversibly damage the part or the whole entity, and how likely it is to happen, etc.
   - Format: "Normal 0 / Emergency X (comprehensive and grounded reasoning why you annotate this certain level)"

6. **Recipient Condition:** What attributes must the recipient have?
   - Every affordance must have a recipient. It can be the object, person, or material that this part acts upon
   - Define scope and limits using attribute categories (shape, size, rigidity, durability, surface, etc.)
   - Be fine-grained and comprehensive
   - Example: "thin to medium thickness, soft to semi-rigid rigidity, not harder than glass"

7. **Example Recipient:** List 3-4 concrete examples
   - Must satisfy the recipient condition
   - The proposed recipient must be concrete and specific things or objects that can be easily found in real life, rather than abstract concepts or ideas
   - Choose diverse examples reflecting affordance scope

8. **Failure Case:** When will this affordance NOT work?
   - Comprehensively consider all failure situations:
     * Use condition failures (can't access/prepare the part)
     * Environment condition failures (lack of necessary environmental factors)
     * Recipient condition failures (recipient too hard/thick/incompatible)
     * Action condition failures (user lacks skill/force/precision)
     * Other practical limitations
   - Be specific and realistic

**Requirements:**
- Generate 6 diverse and non-overlapping affordances
- Generate exactly one affordance for each level (Normal 0 and Emergency 1-5)
- Keep everything plausible and grounded—realistic physical-world uses only
- Each affordance should be completable solely by this part of the entity, NOT relying on other parts or the whole entity
- Do not introduce attributes not provided above, and just focus on this part's attributes, NOT other parts' attributes or the whole entity's attributes
- Every affordance must act upon a recipient; please skip any purely decorative, passive, or static role
- Describe the affordance at a high, scenario-agnostic level; omit uses that are too rare or uncommon to be practically meaningful
- If an affordance requires specific conditions (use, environment, recipient), those conditions must be explicitly captured in the corresponding fields

**Schema:**
```json
[
  {{
    "use_condition": "...",
    "environment_condition": "...",
    "attribute": [["attribute statement", "physical/state", "visual/text", "explanation why visual/text"], ...],
    "affordance": "...",
    "level": "Normal 0 / Emergency 1-5 (reason)",
    "recipient_condition": "...",
    "example_recipient": "...",
    "failure_case": "..."
  }}
]
```

**Example (glass lens, part of glasses entity - visible, free, dry, transparent):**
```json
[
  {{
    "use_condition": "NA",
    "environment_condition": "NA",
    "attribute": [
      ["the material of the lens is glass", "physical", "visual", "Glass is visually identifiable by transparency and clarity"],
      ["the shape is round and flat disc", "physical", "visual", "Circular flat shape is directly observable"],
      ["the rigidity is very rigid", "physical", "text", "Rigidity requires text description to confirm as there is no visual signal to indicate rigidity"]
    ],
    "affordance": "redirect, focus, or magnify light for vision correction or reading",
    "level": "Normal 0 (original intended function of the lens in the glasses)",
    "recipient_condition": "light must pass through; viewer needs magnification; text or objects at appropriate focal distance",
    "example_recipient": "printed text in books, small labels, fine details on objects, digital screens",
    "failure_case": "Fails if lens is dirty or scratched (blocks light), wrong prescription (incorrect magnification), insufficient lighting, or recipient is beyond focal range"
  }},
  {{
    "use_condition": "If we break the glass lens into pieces first",
    "environment_condition": "NA",
    "attribute": [
      ["the material of the lens is glass", "physical", "visual", "Glass is visually identifiable by transparency and clarity"],
      ["the durability is fragile", "physical", "text", "Fragility isn't directly visible, requiring text description to confirm"],
      ["the edge becomes sharp after breaking", "physical", "visual", "The sharpness after of broken glasses is commonsense knowledge, so no need to mention it through text description"]
    ],
    "affordance": "cut, scrape, or pierce small items",
    "level": "Emergency 2 (rarely used due to danger; people rarely break the glasses lens only for cutting purposes; it needs get the lens out first and break it into pieces is irreversible, make itself and the whole glassess not usable anymore; only when no proper cutting tool available)",
    "recipient_condition": "thin to medium thickness, soft to semi-rigid rigidity, not harder than glass, not highly abrasive",
    "example_recipient": "tape, paper, thin plastic wrap, soft fruit skin",
    "failure_case": "Fails if recipient too hard (damages edge), too thick (can't penetrate), or user can't safely handle sharp glass (risk of cuts); fails if pieces too small to grip"
  }},
  ...
]
```

First provide reasoning about affordances. Then output JSON array of 6 diverse and non-overlapping affordances of different levels following the schema and instructions above."""

def format_attrs(attrs):
    return "\n".join([f"  - {k}: {v}" for k, v in attrs.items() if k != "summary"])

def main():
    input_file = "outputs/6_state_combined.json"
    output_file = "outputs/7_functional_affordance.json"
    
    entities = load_json(input_file)
    existing = load_json(output_file) if os.path.exists(output_file) else []
    completed_keys = {(e["entity_name"], e["part_name"], str(e["physical_attributes"]), str(e["state_attributes"])) 
                      for e in existing}
    
    tasks = [(e, p) for e in entities for p in e["parts"] 
             if (e["entity_name"], p["part_name"], str(p["physical_attributes"]), str(p["state_attributes"])) 
             not in completed_keys]
    
    total_parts = sum(len(e["parts"]) for e in entities)
    print(f"Total parts: {total_parts} | Completed: {len(existing)} | Pending: {len(tasks)}")
    
    if not tasks:
        print("All parts already processed")
        return
    
    prompts = [PROMPT.format(entity_name=e["entity_name"], part_name=p["part_name"],
               physical_attrs=format_attrs(p["physical_attributes"]),
               state_attrs=format_attrs(p["state_attributes"]),
               scenario=e["scenario"]) for e, p in tasks if e["scenario"] not in 
               ["waiting_room", "hospital_room", "operating_room", "dental_office", "cafeteria", \
                 "bar", "grocery_store", "warehouse", "recording_studio", "theater_backstage"]]
    
    print(f"Total prompts: {len(prompts)} after filtering out certain scenarios.")
    prompts = prompts[:1024]
    
    results = existing.copy()
    last_saved = [0]  # mutable counter: how many results were saved in previous callbacks

    def save_callback(data):
        # data is the cumulative sorted list; slice to get only the newly arrived results
        new_data = data[last_saved[0]:]
        offset = last_saved[0]
        new = [{"entity_name": tasks[offset + i][0]["entity_name"], "scenario": tasks[offset + i][0]["scenario"],
                "part_name": tasks[offset + i][1]["part_name"], "connected_to": tasks[offset + i][1]["connected_to"],
                "connection": tasks[offset + i][1]["connection"], "physical_attributes": tasks[offset + i][1]["physical_attributes"],
                "state_attributes": tasks[offset + i][1]["state_attributes"],
                "functional_affordances": r["data"]} for i, r in enumerate(new_data) if r and "data" in r]
        save_json_append(new, output_file)
        last_saved[0] = len(data)
        print(f"\nSaved: {len(new)} new results, total so far: {len(existing) + len(data)}")
    
    batch_results = batch_generate(prompts, max_workers=1024, save_callback=save_callback, save_interval=50, temperature=0.7)
    
    for i, r in enumerate(batch_results):
        if r and "data" in r:
            e, p = tasks[i]
            results.append({"entity_name": e["entity_name"], "scenario": e["scenario"],
                           "part_name": p["part_name"], "connected_to": p["connected_to"],
                           "connection": p["connection"], "physical_attributes": p["physical_attributes"],
                           "state_attributes": p["state_attributes"],
                           "functional_affordances": r["data"]})
    
    save_json(results, output_file)
    success_rate = (len(results) - len(existing)) / len(tasks) * 100 if tasks else 100
    print(f"\nCompleted: {len(results)-len(existing)}/{len(tasks)} ({success_rate:.1f}%) | Total: {len(results)}")

if __name__ == "__main__":
    main()
