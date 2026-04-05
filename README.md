<p align="center">
  <img src="assets/logo.jpg" alt="CreativityBench logo" width="40">
  <img src="assets/logo.jpg" alt="CreativityBench logo" width="60">
  <img src="assets/logo.jpg" alt="CreativityBench logo" width="80">
</p>

# CreativityBench: Evaluating Agent Creative Reasoning via Affordance-Based Tool Repurposing

[**📊 Dataset**](https://drive.google.com/file/d/1nQ1p2S4dxTBVZfAvlCW7YPLxztRtPnke/view?usp=sharing) | [**📖 Paper**](TBD) 
<!-- | [**🤗 Model**](TBD) -->

![CreativityBench](assets/intro.png)

CreativityBench is a benchmark for evaluating creative reasoning through affordance-based tool repurposing. The repository contains the full pipeline for:

- building affordance-rich entity annotations,
- constructing benchmark tasks from those annotations,
- evaluating model predictions and judging the results.

## 🗂️ Repository Structure

```text
CreativityBench-Open/
├── dataset/
├── annotation/
├── task_creation/
├── evaluation/
├── assets/
├── requirements.txt
└── README.md
```

- `dataset/`: sample task file for evaluation
- `annotation/`: generate partonomy, physical attributes, state attributes, and functional affordances
- `task_creation/`: turn annotated entities into benchmark tasks
- `evaluation/`: run model inference and judge predictions

Each subfolder has its own concise README with folder-specific details.

## ⚙️ Installation

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The current requirements cover the code in `annotation/`, `task_creation/`, and `evaluation/`.

## 📦 Data

Before using the released benchmark data, first download the dataset from the link below:

- Newest Version: [🔗 Download](https://drive.google.com/file/d/1C7PqRknVjdj4OBpVPT22Opo167lHNyjs/view?usp=sharing)

The link above leads to our release of the first batch of 3.3K tasks. This repository now only includes a sample task file by default:

- `dataset/sample_tasks.json`

If you only want a quick evaluation demo, you can use this sample task file directly. For larger-scale evaluation, point `TASK_FILE` to the full downloaded task file.

### 🧾 Task Format

The sample evaluation file is a JSON list of task objects. Each task currently includes:

- `task_id`: unique task identifier
- `scenario`: scenario label
- `setting`: metadata such as difficulty and entity count
- `golds`: gold entity-part affordance reference
- `entities`: candidate entities and their descriptions
- `items`: extra scene items
- `environment`: natural-language scene description
- `task`: user-facing task description
- `solution`: gold solution steps

## 🚀 Pipeline

Run the benchmark pipeline in this order:

### 1. 🧩 Annotation

Generate structured entity annotations from seed entities:

```bash
cd annotation
./run.sh
```

This stage starts from `1_sample_entities.json` and produces:

- partonomy graphs,
- physical and state variants,
- part-level functional affordances,
- final assembled entities in `annotation/outputs/`.

### 2. 🛠️ Task Creation

Build benchmark tasks from the annotation outputs:

```bash
cd task_creation
./run.sh
```

This stage clusters affordances, samples gold examples and alternatives, and writes benchmark tasks to `task_creation/outputs/`.

### 3. 🔎 Evaluation

Run model inference on the benchmark tasks and judge the outputs:

```bash
cd evaluation
./run.sh
```

By default, `evaluation/run.sh` runs both:

- `evaluate.py`
- `judge.py`

Set `RUN_JUDGE=0` if you only want raw model outputs.

The evaluation source file is changeable through `TASK_FILE`. The bundled sample file is only a default example, and larger downloaded task files can be swapped in later without changing the evaluation code.

## 🤖 Model Configuration

The repository uses environment variables for model and API configuration. Do not hard-code credentials in the codebase.

OpenAI:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Optional:

```bash
export OPENAI_MODEL="gpt-5.2"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

For `evaluation/`, local or hosted vLLM-compatible endpoints are also supported:

```bash
export VLLM_API_KEY="EMPTY"
export VLLM_BASE_URL="http://localhost:8000/v1"
```

## 📝 Notes

- `annotation/` and `task_creation/` use OpenAI-compatible API calls.
- `evaluation/` currently supports OpenAI and vLLM-compatible model endpoints.
- `dataset/sample_tasks.json` is only a bundled sample input for evaluation, and `TASK_FILE` can point to any compatible task JSON.
- All stage runners are relative-path based and avoid machine-specific hard-coded `.env` sourcing.

## 📚 Citation
```
@article{qian2026creativitybench,
  title={CreativityBench: Evaluating Agent Creative Reasoning via Affordance-Based Tool Repurposing},
  author={Qian, Cheng and Ha, Hyeonjeong and Liu, Jiayu and He, Bingxiang and Kim, Jeonghwan and Liu, Jiateng and Li, Bingxuan and Tiwari, Aditi and Dalal, Dwip and Wang, Zhenhailong and Chen, Xiusi and Namazifar, Mahdi and Li, Yunzhu and Ji, Heng},
  journal={arXiv preprint arXiv:XXX},
  year={2026}
}
```
