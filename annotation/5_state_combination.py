import random
from itertools import product
from tqdm import tqdm
from utils import load_json, save_json

def generate_combinations(attribute_lists, max_combinations=6):
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
    source_file = "outputs/4_physical_combined.json"
    input_file = "outputs/5_state_attributes.json"
    output_file = "outputs/6_state_combined.json"

    # Build expected parts per entity instance from the physical combined file
    source_data = load_json(source_file)
    expected_parts = {e["entity_name"]: {p["part_name"] for p in e["parts"]} for e in source_data}

    data = load_json(input_file)
    print(f"Loaded {len(data)} part state entries")
    
    entity_instances = {}
    for entry in data:
        entity_name = entry["entity_name"]
        if entity_name not in entity_instances:
            entity_instances[entity_name] = {
                "entity_name": entity_name,
                "scenario": entry["scenario"],
                "parts": {}
            }
        
        entity_instances[entity_name]["parts"][entry["part_name"]] = {
            "physical_attributes": entry["physical_attributes"],
            "state_attributes": entry["state_attributes"],
            "connected_to": entry["connected_to"],
            "connection": entry["connection"]
        }
    
    print(f"Grouped into {len(entity_instances)} entity instances")
    print("Generating state attribute combinations...")
    
    # Generate all combinations first, grouped by base entity name
    base_entity_results = {}
    skipped = []
    for instance_data in tqdm(entity_instances.values(), desc="Combining", unit="instance"):
        entity_name = instance_data["entity_name"]

        # Drop instance if any expected part is missing
        assembled_parts = set(instance_data["parts"].keys())
        missing = expected_parts.get(entity_name, set()) - assembled_parts
        if missing:
            skipped.append((entity_name, missing))
            continue

        # Extract base entity name (remove number suffix like " 1", " 2")
        base_name = entity_name.rsplit(' ', 1)[0] if entity_name[-1].isdigit() else entity_name
        
        if base_name not in base_entity_results:
            base_entity_results[base_name] = []
        
        parts = list(instance_data["parts"].keys())
        attribute_lists = [instance_data["parts"][part]["state_attributes"] for part in parts]
        combinations = generate_combinations(attribute_lists, max_combinations=6)
        
        for combo in combinations:
            entry = {
                "base_name": base_name,
                "scenario": instance_data["scenario"],
                "parts": [
                    {
                        "part_name": parts[i],
                        "connected_to": instance_data["parts"][parts[i]]["connected_to"],
                        "connection": instance_data["parts"][parts[i]]["connection"],
                        "physical_attributes": instance_data["parts"][parts[i]]["physical_attributes"],
                        "state_attributes": combo[i]
                    } for i in range(len(parts))
                ]
            }
            base_entity_results[base_name].append(entry)

    if skipped:
        print(f"\nSkipped {len(skipped)} incomplete entity instances:")
        for name, missing in skipped:
            print(f"  {name}: missing parts — {', '.join(sorted(missing))}")

    # Renumber all variants sequentially for each base entity
    print("Renumbering variants...")
    results = []
    for base_name, variants in tqdm(base_entity_results.items(), desc="Renumbering", unit="entity"):
        for variant_idx, variant in enumerate(variants, 1):
            # Create new entry with entity_name first
            entry = {
                "entity_name": f"{base_name} {variant_idx}",
                "scenario": variant["scenario"],
                "parts": variant["parts"]
            }
            results.append(entry)
    
    save_json(results, output_file)
    print(f"\nGenerated {len(results)} final entities (max 8×6=48 per original). Saved to {output_file}")

if __name__ == "__main__":
    main()
