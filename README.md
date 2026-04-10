# SCD-MMPSR

Semi-Supervised Cross-Domain Learning Framework for Multitask Multimodal Psychological States Recognition.

This repository contains a configurable training pipeline for multimodal psychological state recognition across multiple datasets. The project combines several input modalities, builds cached feature representations with pretrained encoders, fuses them in a multitask model, and predicts:

- emotion recognition,
- personality traits,
- AH / binary presence-absence target.

## What The System Does

At a high level, the pipeline works as follows:

1. Read dataset metadata from CSV files.
2. Find matching video and audio files for each sample.
3. Extract modality-specific embeddings for face, audio, text, and behavior descriptions.
4. Cache extracted features to avoid recomputing them on every run.
5. Merge enabled datasets into a shared training pipeline.
6. Train a multitask fusion model with optional ablations and optional hyperparameter search.
7. Evaluate on dev/test splits and save checkpoints and logs.

The main entry point is [`main.py`](main.py).

## Project Structure

- [`main.py`](main.py): orchestration entry point; loads config, initializes extractors, builds datasets/loaders, launches training or hyperparameter search.
- [`config.toml`](config.toml): main experiment configuration.
- [`search_params.toml`](search_params.toml): search space and defaults for greedy/exhaustive hyperparameter search.
- [`data/`](data): local copies of dataset annotation/metadata CSV files used for convenience and documentation.
- [`src/train.py`](src/train.py): training loop, validation/test evaluation, metric aggregation, checkpointing, early stopping.
- [`src/data_loading/dataset_builder.py`](src/data_loading/dataset_builder.py): dataset and dataloader creation, split fractions, collate logic.
- [`src/data_loading/dataset_multimodal.py`](src/data_loading/dataset_multimodal.py): sample indexing, label assembly, per-modality feature extraction, feature caching.
- [`src/data_loading/pretrained_extractors.py`](src/data_loading/pretrained_extractors.py): pretrained encoders for face/video, audio, text, and behavior modalities.
- [`src/models/models.py`](src/models/models.py): multitask fusion architectures and ablation-aware variants.
- [`src/utils/feature_store.py`](src/utils/feature_store.py): cache storage for extracted features and metadata.

## Local Annotation Files

The repository includes a local `data/` directory with split annotation tables prepared for this project:

```text
data/
  cmu_mosei/
    train_full_with_description.csv
    dev_full_with_description.csv
    test_full_with_description.csv
  fiv2/
    train_full_with_description.csv
    dev_full_with_description.csv
    test_full_with_description.csv
  bah/
    train_full_with_description.csv
    dev_full_with_description.csv
    test_full_with_description.csv
```

These files store annotation and metadata tables used by the training pipeline. They do not replace the original raw datasets and do not include the source video or audio files.

## Supported Datasets

The current configuration supports three datasets:

- `cmu_mosei`: emotion labels.
- `fiv2`: personality labels.
- `bah`: AH labels.

Each dataset is configured independently in [`config.toml`](config.toml) through:

- `base_dir`
- `csv_path`
- `video_dir`
- `audio_dir`
- `train_fraction`, `dev_fraction`, `test_fraction`

The training loader is built as a concatenation of enabled training subsets from all configured datasets.

## Dataset Provenance, Access, and Ethics

This project does not create, own, or redistribute the original source datasets. It relies on previously released third-party research datasets and project-specific CSV annotations prepared for this pipeline.

When using this repository, please make sure that your use of each dataset complies with its original license, access policy, consent terms, and any institutional or ethics requirements that may apply.

Official sources for the datasets used here:

- `CMU-MOSEI`: official paper: https://aclanthology.org/P18-1208/ ; CMU MultiComp / SDK resources: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
- `First Impressions V2`: official ChaLearn dataset page: https://chalearnlap.cvc.uab.cat/dataset/24/description/
- `BAH`: official dataset page: https://liviaets.github.io/bah-dataset/ ; paper page: https://openreview.net/forum?id=jYDHVscRO3

Notes on access conditions:

- `CMU-MOSEI` is a public research dataset described by CMU MultiComp.
- `First Impressions V2` is publicly documented through ChaLearn and may require registration/sign-in to access the files.
- `BAH` is publicly documented, but the dataset page states that access is provided under a research-only license and requires following the authors' request instructions.

The demographic or sensitive attributes available in the source datasets, such as age, sex, nationality, or ethnicity where applicable, originate from the original dataset providers. This repository does not create new demographic annotations and does not claim authorship over those labels.

## Expected Input Data

Each dataset split is expected to provide:

- a CSV file with sample metadata,
- a directory with video files,
- a directory with audio files.

In this repository, the [`data/`](data) folder stores local copies of the annotation CSV files only. It does not contain the raw video or audio data.

The code expects a `video_name` column in each CSV. Depending on the target task and enabled modalities, the CSV should also contain:

- emotion columns for `cmu_mosei`: `Neutral`, `Anger`, `Disgust`, `Fear`, `Happiness`, `Sadness`, `Surprise`
- personality columns for `fiv2`: `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `non-neuroticism`
- AH columns for `bah`: `absence_full`, `presence_full`
- text column for text modality: `text`
- behavior-description column for behavior modality: by default `text_llm`, configurable via `dataloader.text_description_column`

Example path pattern from the default config:

```toml
[datasets.cmu_mosei]
base_dir = "/path/to/CMU-MOSEI/"
csv_path = "{base_dir}/{split}_full_with_description.csv"
video_dir = "{base_dir}/video/{split}/"
audio_dir = "{base_dir}/audio/{split}/"
```

## Recommended Dataset Directory Structure

Each dataset should be organized so that the configured CSV file, video directory, and audio directory exist for every split used by the experiment.

Example for one dataset:

```text
DATASET_ROOT/
├── train_full_with_description.csv
├── dev_full_with_description.csv
├── test_full_with_description.csv
├── video/
│   ├── train/
│   │   ├── sample_0001.mp4
│   │   ├── sample_0002.mp4
│   │   └── ...
│   ├── dev/
│   │   └── ...
│   └── test/
│       └── ...
└── audio/
    ├── train/
    │   ├── sample_0001.wav
    │   ├── sample_0002.wav
    │   └── ...
    ├── dev/
    │   └── ...
    └── test/
        └── ...
```

For the default configuration, the repository assumes the following structure pattern for each dataset:

- `base_dir`: dataset root directory
- `{base_dir}/{split}_full_with_description.csv`: metadata for a split
- `{base_dir}/video/{split}/`: video files for that split
- `{base_dir}/audio/{split}/`: audio files for that split

## What Each Data Component Contains

- CSV file: one row per sample, with `video_name` and the labels and text fields required by the enabled tasks and modalities.
- `video/<split>/`: source videos used for face-frame extraction.
- `audio/<split>/`: audio tracks used for audio embedding extraction.

Inside this repository, `data/` is intended for annotation tables such as `train_full_with_description.csv`, `dev_full_with_description.csv`, and `test_full_with_description.csv`.

The loader matches files by `video_name` without requiring a specific extension in the code. It searches for a file whose basename matches the value from the CSV.

For example, if a CSV row contains:

```text
video_name = sample_0001
```

then the loader will look for files such as:

- `video/train/sample_0001.mp4`
- `audio/train/sample_0001.wav`

## Minimal CSV Schema

At minimum, each CSV should contain:

- `video_name`

Depending on which labels and modalities are used, the CSV may additionally need:

- `text`: source text for the text modality
- `text_llm` or another configured description column: source text for the behavior modality
- emotion label columns for `cmu_mosei`
- personality label columns for `fiv2`
- AH label columns for `bah`

## Modalities And Feature Extraction

The system supports four modalities:

- `face`: extracted from video frames after face detection.
- `audio`: extracted from audio files.
- `text`: extracted from the `text` column.
- `behavior`: extracted from the configured description column such as `text_llm`.

Available extractor families in the current codebase include:

- video/face: CLIP-based image encoder
- audio: CLAP or Wav2Vec2-style encoder
- text/behavior: CLIP text, CLAP text, RoBERTa/XLM-R style models, or `michellejieli/emotion_text_classifier`

Feature extraction is configured in the `[embeddings]` section of [`config.toml`](config.toml). Extracted embeddings can be stored in the local feature cache to speed up repeated experiments.

## Model Overview

The fusion model is defined in [`src/models/models.py`](src/models/models.py). The model:

- projects each modality into a shared hidden space,
- optionally applies graph-based interaction between modalities,
- optionally applies cross-attention between task-specific and modality-level representations,
- predicts multiple tasks jointly,
- optionally uses guide-bank representations for task heads.

Implemented model variants:

- `MultiModalFusionModel_v1`
- `MultiModalFusionModel_v2`
- `MultiModalFusionModel_v3`

The default config currently uses `MultiModalFusionModel_v2`.

## Configuration

Most experiment behavior is controlled from [`config.toml`](config.toml):

- `[datasets.*]`: dataset paths and subset fractions
- `[dataloader]`: worker count, shuffle, `prepare_only`, behavior-text column
- `[train.general]`: seed, batch size, epochs, patience, checkpointing, cache saving, device, search mode
- `[train.model]`: model architecture hyperparameters
- `[train.losses]`: multitask and semi-supervised loss settings
- `[train.optimizer]`: optimizer and learning rate
- `[train.scheduler]`: scheduler setup
- `[embeddings]`: extractor choice and embedding aggregation strategy
- `[cache]`: cache behavior and forced re-extraction
- `[ablation]`: module/task/modality ablations

## Installation

Use Python 3.10+ and install dependencies from [`requirements.txt`](requirements.txt).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you plan to use Telegram notifications, also install:

```bash
pip install python-dotenv
```

Then create a `.env` file with:

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Telegram is optional. You can disable it with `use_telegram = false` in [`config.toml`](config.toml).

## How To Run

### 1. Configure dataset paths

Update dataset paths in [`config.toml`](config.toml) so they point to your local copies of `cmu_mosei`, `fiv2`, and `bah`.

### 2. Choose the run mode

The pipeline supports three main modes through `train.general.search_type`:

- `none`: single training run
- `greedy`: greedy hyperparameter search
- `exhaustive`: exhaustive hyperparameter search

### 3. Launch the pipeline

```bash
python main.py
```

## Common Workflows

### Prepare features only

If you want to build caches without starting training:

```toml
[dataloader]
prepare_only = true
```

Then run:

```bash
python main.py
```

### Run a single training experiment

Set:

```toml
[dataloader]
prepare_only = false

[train.general]
search_type = "none"
```

Then run:

```bash
python main.py
```

### Run hyperparameter search

Set `search_type = "greedy"` or `search_type = "exhaustive"` in [`config.toml`](config.toml). Search values are read from [`search_params.toml`](search_params.toml).

## Outputs

Each run creates a timestamped directory under `results/`, for example:

```text
results/results_multimodalfusionmodel_v2_YYYY-MM-DD_HH-MM-SS/
```

The run directory contains:

- `config_copy.toml`: snapshot of the run configuration
- `session_log.txt`: full log output
- `metrics_by_epoch/`: metric logs
- `checkpoints/`: saved best model checkpoints
- `overrides.txt`: search overrides when search mode is used

Cached modality features are stored separately under the path configured by `train.general.save_feature_path`, which defaults to `./features/`.

## Reproducing Results

To reproduce the reported or best configuration results:

1. Install the dependencies listed above.
2. Prepare the datasets with the expected CSV schema and folder layout.
3. Set dataset paths in [`config.toml`](config.toml).
4. Use the best-performing settings in [`config.toml`](config.toml) and, if applicable, [`search_params.toml`](search_params.toml).
5. Run `python main.py`.
6. Collect metrics from the log file and the saved checkpoints in the generated `results/` directory.

If the paper or benchmark section reports a specific best setup, it is recommended to explicitly mark that setup in the config or document it in a dedicated subsection.

## Notes

- The default sample paths in [`config.toml`](config.toml) are local machine paths and should be changed before running the project elsewhere.
- The repository includes `yolov8n-face.pt`, but face extraction also supports MediaPipe-based detection depending on config.
- If a dataset split does not provide a `test` file, the code falls back to the dev loader for test-time evaluation.
