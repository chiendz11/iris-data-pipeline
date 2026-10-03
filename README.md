# Iris data and training pipeline (repo 1/5)

This repository owns the ML side of the lifecycle: deterministic dataset preparation, dataset
fetching, training, offline evaluation, MLflow run logging, immutable model registration, and the
versioned model-result contract. It does not own serving checks, online SLO evaluation, registry
promotion, Kubernetes manifests, or GitOps rendering.

## Why Iris and Logistic Regression?

- Iris has 150 records, four numeric features, three balanced classes, and ships with scikit-learn.
- CI does not depend on Kaggle credentials or an external dataset API, and training takes seconds.
- The dataset contains no PII, so it is safe for a public capstone repository.
- Logistic Regression is small and explainable while consistently exceeding the 0.90 accuracy
  threshold.
- The example remains rich enough to demonstrate DVC versioning, experiment tracking, quality
  gates, model registration, and downstream serving.

Deep learning would increase build time, weight downloads, and compute cost without improving the
MLOps concepts demonstrated by this project.

## Source layout

```text
src/iris_pipeline/
├── data/          # deterministic preparation and event-driven S3 fetch
├── training/      # config, estimator, and training orchestration
├── evaluation/    # offline metrics and champion comparison
└── contracts/     # versioned model-result serialization
tests/
├── unit/
├── training/
└── contract/
contracts/         # consumer-facing JSON Schema
docs/              # contract and ownership documentation
```

Production smoke traffic, canary evaluation, MLflow alias promotion, and desired-state rendering
belong to the release automation image built by `iris-gitops`; none of those capabilities or
credentials are present in the training image.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -e .
REGISTER_MODEL=false MLFLOW_TRACKING_URI=sqlite:///mlflow-local.db dvc repro
cat metrics.json
cat outputs/model-result.json
```

To log and register a model version in a locally running MLflow server:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export REGISTER_MODEL=true
python -m iris_pipeline.data.prepare
python -m iris_pipeline.training.train
```

Training never changes the `champion` alias. It reads the current champion accuracy, applies the
absolute and improvement thresholds from `params.yaml`, tags the newly registered version with the
offline gate result, and writes `outputs/model-result.json` with candidate and baseline model
versions. Online automation is solely responsible for deciding whether to promote that version or
restore the recorded baseline.
Chỉ lỗi MLflow `RESOURCE_DOES_NOT_EXIST` được hiểu là chưa có champion; lỗi server/network hoặc
champion thiếu metric accuracy làm training fail-closed, không đi nhầm vào bootstrap.

## Model-result boundary

The complete, authoritative output is documented in
[`docs/model-output-contract.md`](docs/model-output-contract.md) and validated by
[`contracts/model-result-v1.schema.json`](contracts/model-result-v1.schema.json). It contains model
and dataset provenance plus offline metrics; it contains no manifest path or KServe field.

Three text files are derived from that JSON only to support Argo DAG `when` expressions:

- `outputs/model-version.txt`
- `outputs/baseline-model-version.txt`
- `outputs/quality-gate.txt`
- `outputs/bootstrap-required.txt`

The GitOps-owned dispatcher reads the full JSON from the shared workflow volume and turns it into a
release intent. Consequently, this repo does not authenticate a GitHub App, open a cross-repository
pull request, or know how production manifests are organized.

## DVC, S3, and dataset events

`.dvc/config` contains a placeholder remote. After the infrastructure stack has published the DVC
bucket as a GitHub Environment variable, CI configures the effective S3 URL without committing an
environment-specific bucket name.

For local use:

```bash
export DVC_BUCKET=your-dvc-bucket
./scripts/configure_dvc_remote.sh
dvc push
```

On a merge to `main`, GitHub Actions uses OIDC to obtain short-lived AWS credentials. It builds and
scans the training image, publishes an immutable source-commit tag to ECR, resolves the registry
digest, signs that digest keylessly with Cosign, pushes the DVC cache, and uploads the versioned
dataset object under `datasets/<source-sha>/<training-image-digest>/iris.csv`. The S3 Sensor derives
both the exact `repository@sha256:...` training image and source commit from that immutable key, so a
later tag cannot change which code trains the dataset. The S3 event starts the cluster-owned
training workflow. App CI neither calls GitOps nor mutates Kubernetes.

Training writes the full `model-result-v1` contract to the shared Argo volume. The GitOps-owned
release automation validates that contract, then emits a separate model-release intent for canary,
promote or rollback. This repo never emits workload-release intent because the training image is an
input of the event-driven DAG, not a long-running Kubernetes workload.

Required GitHub Environment `prod` variables are `AWS_REGION`, `AWS_DEPLOY_ROLE_ARN`,
`TRAINING_ECR_REPOSITORY`, and `DVC_BUCKET`. There are no GitHub App keys or long-lived AWS access
keys in this repository.
