import random
from itertools import product
from tqdm import tqdm
from utils import load_json, save_json

def generate_combinations(attribute_lists, max_combinations=8):
    total = 1
    for lst in attribute_lists:
        total *= len(lst)
    
    if total <= max_combinations:
        return list(product(*attribute_lists))
    
    # Shuffle each part's attribute list
    shuffled_lists = [lst.copy() for lst in attribute_lists]
    for lst in shuffled_lists:
        random.shuffle(lst)
    
    combinations = []
    max_length = max(len(lst) for lst in shuffled_lists)
    
    # Take one from each position to ensure all attributes appear
    for pos in range(max_length):
        combo = []
        for part_idx, lst in enumerate(shuffled_lists):
            if pos < len(lst):
                combo.append(lst[pos])
            else:
                # Already used all, randomly sample
                combo.append(random.choice(attribute_lists[part_idx]))
        combinations.append(combo)
        if len(combinations) >= max_combinations:
            break
    
    # If we have more than max, randomly sample down
    if len(combinations) > max_combinations:
        combinations = random.sample(combinations, max_combinations)
    
    return combinations

def main():
    source_file = "outputs/2_partonomy.json"
    input_file = "outputs/3_physical_attributes.json"
    output_file = "outputs/4_physical_combined.json"

    # Build expected parts per entity from the partonomy
    source_data = load_json(source_file)
    expected_parts = {e["entity_name"]: set(e["parts"]) for e in source_data}

    data = load_json(input_file)
    print(f"Loaded {len(data)} part attribute entries")
    
    entities = {}
    for entry in data:
        entity_name = entry["entity_name"]
        if entity_name not in entities:
            entities[entity_name] = {
                "entity_name": entity_name,
                "scenario": entry["scenario"],
                "parts": {}
            }
        entities[entity_name]["parts"][entry["part"]] = {
            "physical_attributes": entry["physical_attributes"],
            "connected_to": entry["connected_to"],
            "connection": entry["connection"]
        }
    
    print(f"Grouped into {len(entities)} entities")
    print("Generating physical attribute combinations...")
    
    results = []
    skipped = []
    for entity_name, entity_data in tqdm(entities.items(), desc="Combining", unit="entity"):
        assembled_parts = set(entity_data["parts"].keys())
        missing = expected_parts.get(entity_name, set()) - assembled_parts
        if missing:
            skipped.append((entity_name, missing))
            continue

        parts = list(entity_data["parts"].keys())
        attribute_lists = [entity_data["parts"][part]["physical_attributes"] for part in parts]
        combinations = generate_combinations(attribute_lists, max_combinations=8)
        
        for combo_idx, combo in enumerate(combinations, 1):
            entry = {
                "entity_name": f"{entity_name} {combo_idx}",
                "scenario": entity_data["scenario"],
                "parts": [
                    {
                        "part_name": parts[i],
                        "connected_to": entity_data["parts"][parts[i]]["connected_to"],
                        "connection": entity_data["parts"][parts[i]]["connection"],
                        "physical_attributes": combo[i]
                    } for i in range(len(parts))
                ]
            }
            results.append(entry)

    if skipped:
        print(f"\nSkipped {len(skipped)} incomplete entities:")
        for name, missing in skipped:
            print(f"  {name}: missing parts — {', '.join(sorted(missing))}")

    save_json(results, output_file)
    print(f"\nGenerated {len(results)} combined entities. Saved to {output_file}")

if __name__ == "__main__":
    main()
