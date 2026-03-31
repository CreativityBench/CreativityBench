# Task Creation Pipeline

This folder builds benchmark tasks from the annotated entities produced by the `annotation/` pipeline. It clusters part affordances, samples gold examples and competing entities, then generates final task records.

Run everything from `task_creation/`. Outputs are written to `task_creation/outputs/`.

## Files

- `1_tight_clustering.py`: Embed affordances and cluster them by scenario.
- `2_sample_compare.py`: Sample gold affordances, generate candidate tasks, and compare other parts against the gold.
- `3_task_creation.py`: Build final benchmark task records from comparison results.
- `utils.py`: Shared OpenAI, JSON, JSONL, embeddings, and parallel-processing utilities.
- `run.sh`: Run the full pipeline.

## Dependency

This pipeline depends on the final annotation output:

- `../annotation/outputs/8_entities_complete.json`

The loader in `utils.py` accepts both standard JSON and JSONL, so it can read the current annotation output format directly.

## Outputs

The pipeline writes:

- `outputs/1_affordance_lookup.json`
- `outputs/1_entity_lookup.json`
- `outputs/1_embeddings.json`
- `outputs/1_clusters.json`
- `outputs/1_centroids.json`
- `outputs/1_stats.json`
- `outputs/embedding_figures/`
- `outputs/2_comparisons.json`
- `outputs/3_tasks_ne.json`

## Pipeline

1. `1_tight_clustering.py`: load annotated entities, embed affordance text, and build scenario-specific clusters
2. `2_sample_compare.py`: sample gold affordances by level and cluster size, generate benchmark situations, and judge whether other parts can solve the same task
3. `3_task_creation.py`: assemble final task entries with entities, scene items, environment text, and solution references

## Setup

From the repo root:

```bash
pip install -r requirements.txt
```

Or from `task_creation/`:

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

Note:

- `1_tight_clustering.py` uses `text-embedding-3-large` for embeddings.

## Run

Full pipeline:

```bash
cd task_creation
./run.sh
```

Run step-by-step:

```bash
cd task_creation
python 1_tight_clustering.py
python 2_sample_compare.py
python 3_task_creation.py
```

## Notes

- `run.sh` creates `outputs/` automatically and requires `OPENAI_API_KEY`.
- Stage 1 caches embeddings, clusters, centroids, and stats under `outputs/`.
- Stages 2 and 3 resume from existing output files when possible.
- Stage 1 currently processes the scenarios listed inside `1_tight_clustering.py`.
