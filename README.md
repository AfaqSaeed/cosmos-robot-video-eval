# cosmos-robot-video-eval

`cosmos-robot-video-eval` is a reproducible Python research demo for generating and evaluating short robotics-oriented synthetic videos with NVIDIA NIM / Cosmos APIs.

The core research question is:

> Are world-model-generated robot videos temporally stable, semantically consistent, physically plausible, and useful enough for downstream robotics data generation?

The project is intentionally small and modular. It can call a configured NVIDIA endpoint when credentials are present, but the full extraction, evaluation, reporting, and Streamlit dashboard workflow also runs on any local MP4 files placed in `data/generated/`.

## Research Motivation

Robot video generation should not be judged only by visual realism. For robotics data generation, a clip also needs stable motion, prompt alignment, plausible physical continuity, and enough quality to be useful for downstream experiments.

This MVP evaluates four groups of signals:

- Temporal stability: frame differences, flicker, optical-flow magnitude stability, and a normalized smoothness score.
- Semantic consistency: optional OpenCLIP prompt/frame similarity and embedding drift, with deterministic mock metrics when OpenCLIP is unavailable.
- Physical plausibility: rule-based checks for jumps, flicker, unstable optical flow, and likely camera discontinuities.
- Task usefulness: proxy checks for readability, frame count, motion stability, and semantic consistency when available.

The aggregate score is:

```text
overall_score =
0.35 * temporal_score +
0.25 * semantic_score +
0.25 * physical_plausibility_score +
0.15 * task_usefulness_score
```

## Repository Layout

```text
configs/                 Prompt and evaluation configs
data/generated/          Generated or local sample MP4 videos
data/frames/             Extracted frames and frame metadata
data/metrics/            Per-video metrics JSON and summary CSV
data/reports/            Markdown reports
scripts/                 CLI entry points
src/generation/          NVIDIA NIM client and Cosmos generator
src/preprocessing/       Video metadata and frame extraction
src/evaluation/          Temporal, semantic, physical, task, aggregate metrics
src/dashboard/           Streamlit dashboard
src/reporting/           Markdown report generation
tests/                   Pytest coverage for core MVP behavior
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Environment Variables

Do not hardcode API keys. Set these in your shell or in a local `.env` file:

```text
NVIDIA_API_KEY=your_api_key
NVIDIA_BASE_URL=https://your-nvidia-base-url
NVIDIA_COSMOS_ENDPOINT=/v1/video/generations
```

The local `.env` file is ignored by Git. `python-dotenv` loads it automatically when present, and real shell or CI environment variables take precedence. `NVIDIA_COSMOS_ENDPOINT` is shown as an example in `.env.example`; the generator primarily uses the endpoint in `configs/prompts_robotics.yaml`.

Create your local file:

```bash
cp .env.example .env
```

On Windows:

```powershell
copy .env.example .env
```

Then edit `.env` locally and paste your real NVIDIA NIM key there. Do not commit `.env`.

## Run Without an NVIDIA API Key

Place one or more `.mp4` files in `data/generated/`, then run:

```bash
python scripts/02_extract_frames.py --video_dir data/generated
python scripts/03_evaluate_videos.py --video_dir data/generated
streamlit run src/dashboard/app.py
python scripts/05_make_report.py
```

This path does not require `NVIDIA_API_KEY`. Semantic metrics use deterministic mock values if `open_clip` is not installed, and those metrics are marked as `unavailable_mock`.

## Run With NVIDIA NIM / Cosmos API

Inspect the generation payloads first:

```bash
python scripts/01_generate_videos.py --config configs/prompts_robotics.yaml --dry-run
```

After setting `NVIDIA_API_KEY` and `NVIDIA_BASE_URL`, generate videos:

```bash
python scripts/01_generate_videos.py --config configs/prompts_robotics.yaml
```

Then evaluate and view:

```bash
python scripts/02_extract_frames.py --video_dir data/generated
python scripts/03_evaluate_videos.py --video_dir data/generated
streamlit run src/dashboard/app.py
python scripts/05_make_report.py
```

## Example Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python scripts/01_generate_videos.py --config configs/prompts_robotics.yaml --dry-run
python scripts/01_generate_videos.py --config configs/prompts_robotics.yaml
python scripts/02_extract_frames.py --video_dir data/generated
python scripts/03_evaluate_videos.py --video_dir data/generated
streamlit run src/dashboard/app.py
python scripts/05_make_report.py
```

For Windows activation:

```powershell
.venv\Scripts\activate
```

## Testing

```bash
pytest
```

The tests create small synthetic videos and frames locally. They do not call NVIDIA APIs.

## Notes for Extension

- Replace or extend the endpoint payload in `CosmosGenerator.build_payload` when your Cosmos/NIM deployment expects a different request schema.
- Add an external NVIDIA Cosmos Reasoner or VLM call behind `PhysicalReasoningEvaluator` without changing the aggregation contract.
- Install `open_clip_torch` and a compatible Torch build to enable real semantic metrics.
- Keep local sample videos in `data/generated/` to make demos reproducible when credentials are unavailable.
