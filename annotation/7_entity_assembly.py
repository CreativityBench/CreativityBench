import re
from tqdm import tqdm
from utils import load_json, save_json, save_json_readable

SCENARIOS = [
    "kitchen", "living_room", "bedroom", "bathroom", "garage", "home_office",
    "dining_room", "garden", "basement",
    "classroom", "chemistry_lab", "physics_lab", "biology_lab", "computer_lab",
    "art_studio", "music_room", "photography_studio", "library_room", "gym_room",
    "meeting_room", "executive_office", "storage_room", "waiting_room",
    "hospital_room", "operating_room", "dental_office",
    "cafeteria", "bar", "grocery_store", "warehouse",
    "recording_studio", "theater_backstage"
]

def clean_scenario(s):
    for scenario in SCENARIOS:
        if scenario in s:
            return scenario
    return s

def main():
    source_file = "outputs/6_state_combined.json"
    input_file = "outputs/7_functional_affordance.json"
    output_file = "outputs/8_entities_complete.json"

    # Build expected parts per entity from the original source
    source_data = load_json(source_file)
    expected_parts = {e["entity_name"]: {p["part_name"] for p in e["parts"]} for e in source_data}

    parts_data = load_json(input_file)
    print(f"Loaded {len(parts_data)} part affordances from {len(expected_parts)} source entities")

    # Group parts by entity
    entities = {}
    for part_entry in tqdm(parts_data, desc="Grouping", unit="part"):
        entity_name = part_entry["entity_name"]
        if entity_name not in entities:
            entities[entity_name] = {
                "entity_name": entity_name,
                "scenario": part_entry["scenario"],
                "parts": []
            }
        entities[entity_name]["parts"].append({
            "part_name": part_entry["part_name"],
            "connected_to": part_entry["connected_to"],
            "connection": part_entry["connection"],
            "physical_attributes": part_entry["physical_attributes"],
            "state_attributes": part_entry["state_attributes"],
            "functional_affordances": part_entry["functional_affordances"]
        })

    # Drop entities with missing parts
    results = []
    skipped = []
    for entity_name, entity_data in entities.items():
        print(entity_name)
        assembled_parts = {p["part_name"] for p in entity_data["parts"]}
        missing = expected_parts.get(entity_name, set()) - assembled_parts
        if missing:
            skipped.append((entity_name, missing))
        else:
            entity_data["scenario"] = clean_scenario(entity_data["scenario"])
            results.append(entity_data)

    if skipped:
        print(f"\nSkipped {len(skipped)} incomplete entities:")
        for name, missing in skipped:
            print(f"  {name}: missing parts — {', '.join(sorted(missing))}")
    
    # save_json(results, output_file)
    save_json_readable(results, output_file.replace(".json", "_readable.json"))
    print(f"\nAssembled {len(results)} complete entities from {len(parts_data)} parts")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
