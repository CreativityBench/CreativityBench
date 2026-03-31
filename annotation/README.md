# Annotation Pipeline

This folder converts a JSONL list of seed entities into structured CreativityBench annotations: partonomy, physical attributes, state attributes, functional affordances, and final assembled entities.

Run everything from `annotation/`. Outputs are written to `annotation/outputs/`.

## Files

- `1_sample_entities.json`: Input JSONL file with `name` and `scenario`.
- `1_partonomy_graph.py`: Generate parts and relations for each entity.
- `2_physical_attributes.py`: Generate physical variants for each part.
- `3_physical_combination.py`: Combine physical variants into entity variants.
- `4_state_attributes.py`: Generate state variants for each part instance.
- `5_state_combination.py`: Combine state variants into entity variants.
- `6_functional_affordance.py`: Generate functional affordances for each part instance.
- `7_entity_assembly.py`: Assemble complete entities from part-level outputs.
- `utils.py`: Shared OpenAI, JSONL, and parallel-processing utilities.
- `run.sh`: Run the full pipeline.

## Input

`1_sample_entities.json` is JSONL, one object per line:

```json
{"name": "acacia wood end-grain cutting board", "scenario": "kitchen"}
{"name": "tufted three-seat sofa with chaise", "scenario": "living_room"}
```

## Outputs

The pipeline writes:

- `outputs/2_partonomy.json`
- `outputs/3_physical_attributes.json`
- `outputs/4_physical_combined.json`
- `outputs/5_state_attributes.json`
- `outputs/6_state_combined.json`
- `outputs/7_functional_affordance.json`
- `outputs/8_entities_complete.json`

## Pipeline

1. `1_partonomy_graph.py`: entity -> parts and relations
2. `2_physical_attributes.py`: part -> multiple physical variants
3. `3_physical_combination.py`: entity -> up to 8 physical combinations
4. `4_state_attributes.py`: part instance -> multiple state variants
5. `5_state_combination.py`: entity instance -> up to 6 state combinations
6. `6_functional_affordance.py`: part instance -> 6 affordances (`Normal 0`, `Emergency 1-5`)
7. `7_entity_assembly.py`: part-level outputs -> complete entities

At most, stages 3 and 5 can produce `8 x 6 = 48` variants per seed entity.

## Setup

From the repo root:

```bash
pip install -r requirements.txt
```

Or from `annotation/`:

```bash
pip install -r ../requirements.txt
```

## OpenAI Configuration

The scripts read configuration from environment variables. Do not hard-code keys in the repo.

Required:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Optional:

```bash
export OPENAI_MODEL="gpt-5.2"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Current defaults in code:

- `OPENAI_MODEL` defaults to `gpt-5.2`
- `OPENAI_BASE_URL` defaults to the standard OpenAI API endpoint

## Run

Full pipeline:

```bash
cd annotation
./run.sh
```

Run step-by-step:

```bash
cd annotation
python 1_partonomy_graph.py
python 2_physical_attributes.py
python 3_physical_combination.py
python 4_state_attributes.py
python 5_state_combination.py
python 6_functional_affordance.py
python 7_entity_assembly.py
```

## Notes

- Files are JSONL, one object per line.
- `run.sh` creates `outputs/` automatically and requires `OPENAI_API_KEY`.
- The generation stages resume from existing outputs when possible.
- `6_functional_affordance.py` currently filters some scenarios and caps prompt generation at 1024 items, so review that file before large-scale runs.
