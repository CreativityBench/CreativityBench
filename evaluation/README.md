# Evaluation

This folder runs model inference on benchmark tasks and then judges the predictions against the gold annotations and solutions.

Run everything from `evaluation/`.

## Files

- `evaluate.py`: run model inference on task files
- `judge.py`: score saved predictions with rule-based checks plus an LLM judge
- `models.py`: shared model router for OpenAI and vLLM
- `utils.py`: prompt builders, response parsing, and interactive-mode logic
- `run.sh`: run evaluation and, by default, judging

## Dependency

Evaluation ships with a sample task file by default:

- `../dataset/sample_tasks.json`

Judging also uses:

- `../task_creation/outputs/1_entity_lookup.json`

`TASK_FILE` is changeable. The sample file is only a default example for quick runs, and you can later point evaluation to any larger compatible task JSON file.

## Model Providers

Provider routing is based on the model name prefix unless `MODEL_PROVIDER` is set:

- `gpt-*` -> OpenAI
- anything else -> vLLM-compatible OpenAI API

Required environment variables depend on the provider:

OpenAI:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

vLLM:

```bash
export VLLM_API_KEY="EMPTY"
export VLLM_BASE_URL="http://localhost:8000/v1"
```

Optional:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export MODEL_PROVIDER=""
```

Do not hard-code keys in this folder.

## Evaluate

Main environment variables in `evaluate.py`:

- `TASK_FILE`: input task JSON file, default `../dataset/sample_tasks.json`
- `OUTPUT_DIR`: directory for raw model outputs
- `MODEL`: model name, default `gpt-5.2`
- `MODE`: `static`, `cot`, or `interactive`
- `MAX_WORKERS`: parallel worker count
- `SAVE_EVERY`: checkpoint frequency
- `TEMPERATURES`: comma-separated list, default `0.0,0.3,0.7`
- `MAX_TURNS`: only used by interactive mode, defined in `utils.py`

Run examples:

```bash
cd evaluation
TASK_FILE=../dataset/sample_tasks.json MODEL=gpt-5.2 MODE=static python evaluate.py
TASK_FILE=../dataset/sample_tasks.json MODEL=gpt-5.2 MODE=cot python evaluate.py
TASK_FILE=../dataset/sample_tasks.json MODEL=my-local-model MODE=interactive python evaluate.py
```

Full run:

```bash
cd evaluation
MODEL=gpt-5.2 MODE=static ./run.sh
```

Modes:

- `static`: full entity descriptions are given up front
- `cot`: same input, but requires structured reasoning output
- `interactive`: the model explores one scene item at a time before answering

Raw outputs are saved to:

- `outputs/{task_file_stem}__{model}__{mode}.json`

Each task stores one result per temperature. Interactive results also include turn history.

## Judge

`judge.py` reads model outputs from `OUTPUT_DIR` and writes judged files to `JUDGED_OUTPUT_DIR`.

Main environment variables in `judge.py`:

- `OUTPUT_DIR`: directory containing raw evaluation outputs
- `JUDGED_OUTPUT_DIR`: directory for judged outputs
- `JUDGE_MODEL`: judge model, default falls back to `MODEL` or `gpt-5.2`
- `JUDGE_TEMPERATURE`: default `0.0`
- `JUDGE_MAX_WORKERS`: sample-level parallelism
- `JUDGE_FILE_MAX_WORKERS`: file-level parallelism

Run example:

```bash
cd evaluation
OUTPUT_DIR=./outputs JUDGED_OUTPUT_DIR=./judged_outputs JUDGE_MODEL=gpt-5.2 python judge.py
```

Judging focuses on the zero-temperature sample for each task and combines:

- exact or fuzzy entity/part matching
- relation to the gold answer (`gold`, `similar-*`, or `not similar`)
- LLM-based scoring of condition coverage, grounding, correctness, and feasibility

## Notes

- `run.sh` runs `evaluate.py` first and then `judge.py`. Set `RUN_JUDGE=0` to skip judging.
- The default task source is `../dataset/sample_tasks.json`, but `TASK_FILE` can be changed to any compatible task file.
- `evaluate.py` resumes from existing output files when possible.
- `judge.py` also resumes and skips already judged samples.
