# Netflix title type classifier

## Problem framing

This project predicts whether a Netflix catalog entry is a **Movie** or a **TV Show** from its description, genres, country, content rating, and release year. Catalog operations teams and analysts could use it as a first-pass label for newly ingested or incomplete records, with a person reviewing uncertain or consequential cases. Correct categorization makes a large catalog easier to organize and analyze; this is a catalog-labeling aid, not a recommendation engine or an official Netflix product.

The classes are imbalanced, so model selection uses **5-fold cross-validated macro F1**: it gives equal weight to performance on Movies and TV Shows. **Balanced accuracy** is also reported in cross-validation and on the stratified holdout set. The held-out test data is kept out of preprocessing and tuning.

## Data and preparation

`netflix_titles.csv` is included in the repository. It contains catalog metadata and `type`, the target. The original source is [Kaggle: Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows). To retrieve a fresh copy, configure Kaggle credentials, then run `kaggle datasets download -d shivamb/netflix-shows --unzip -p .` from the project root. Alternatively, download the CSV from the dataset page and place it there.

`model.py` provides the shared, reproducible cleaning and preprocessing functions. Rows without a supported target are removed; missing description, genre, and country values become empty text; missing ratings become `Unknown`; and malformed release years are coerced to missing and imputed within the training pipeline. The selected interpretable inputs are description, genre, country, rating, and release year. The title, cast, director, duration, and show identifier are excluded; in particular, no target-derived features are used.

The notebook explores target and rating distributions, missingness, release-year distributions and IQR outliers, and the correlation between release year and a binary target encoding. EDA and model selection are in `notebook.ipynb`.

## Modeling and evaluation

Training starts with a class-weighted Logistic Regression baseline, then compares it with class-weighted LinearSVC and a class-weighted decision tree. Five-fold stratified cross-validation tunes regularization for the linear models and depth/leaf size for the tree. TF-IDF and imputers are fitted inside each CV pipeline, preventing preprocessing leakage. The highest macro-F1 model is evaluated on a stratified holdout set.

`python train.py` prints the holdout classification report, macro F1, and balanced accuracy. It saves the selected model as `models/netflix_type.joblib`, the best CV score and parameters per model as `models/model_comparison.csv`, and signed linear weights or tree feature importances as `models/feature_importance.csv`. Metrics are generated when training runs; no scores are claimed here before execution.

## Run locally

Use Python 3.11 or later:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

Optional training arguments: `--data PATH`, `--model PATH`, `--cv FOLDS`, and `--n-jobs JOBS`. The notebook can be opened with `jupyter notebook notebook.ipynb` after installing the requirements.

## HTTP API

`serve.py` starts a Flask service on `http://localhost:9696`. Train first, then run:

```bash
python serve.py
```

`GET /health` returns `200 OK`. `POST /predict` accepts a JSON object with optional `description`, `listed_in`, `country`, `rating`, and numeric `release_year` fields and returns a `predicted_type` label:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"description":"A family discovers a mysterious new world.","listed_in":"Children & Family Movies, Fantasy","country":"United States","rating":"PG","release_year":2020}' \
  http://localhost:9696/predict
```

Omitted inputs are treated as unknown/empty. Unsupported fields and invalid value types return HTTP 400; if the model artifact is missing, predictions return HTTP 503. `MODEL_PATH` and `PORT` can override the default artifact path and port. `predict.py` is the reusable artifact-loading and inference module; `serve.py` contains the HTTP routes.

## Docker

The image includes the dataset and trains the model during the image build, so it does not depend on a pre-generated local artifact:

```bash
docker build -t netflix-title-classifier .
docker run --rm -p 9696:9696 netflix-title-classifier
```

Then call the same health and prediction endpoints on port 9696. The container training command uses three CV folds and two parallel jobs to keep build resources bounded; local training defaults to five folds. Docker and a compatible container runtime are required.

## Architecture

```text
netflix_titles.csv -> prepare_data / CV model selection -> models/netflix_type.joblib
                                                            |
Client -> POST /predict -> serve.py -> predict.py -> prediction
Client -> GET  /health  -> serve.py
```

## Limitations and deployment

The catalog is a historical snapshot, so labeling patterns and language may not match new content or other streaming services. Text features can reflect dataset-specific associations, and title descriptions alone may not reliably signal format. Balanced metrics do not replace per-class error review. Treat predictions as suggestions and monitor performance on current, representative data before operational use.

No cloud deployment URL or demo recording is included. The Docker image is ready to deploy to a container platform; deployment, access instructions, and cost teardown steps should be documented once a provider is chosen.
