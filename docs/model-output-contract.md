# Model result contract

`outputs/model-result.json` is the only cross-repository result owned by the training image. It
describes what training produced; it deliberately does not describe a KServe object, a Git path,
a canary percentage, or a promotion operation.

The JSON must validate against [`contracts/model-result-v1.schema.json`](../contracts/model-result-v1.schema.json).
An example registered result is:

```json
{
  "model_name": "iris-classifier",
  "model_version": "12",
  "baseline_model_version": "11",
  "quality_gate_passed": true,
  "bootstrap_required": false,
  "run_id": "abc123",
  "dataset_version": "etag-456",
  "source_repository": "chiendz11/iris-data-pipeline",
  "source_sha": "d34db33f",
  "metrics": {
    "accuracy": 0.9667,
    "f1_macro": 0.9666,
    "baseline_accuracy": 0.95
  },
  "contract_version": "v1"
}
```

`model-version.txt`, `baseline-model-version.txt`, `quality-gate.txt`, and
`bootstrap-required.txt` are derived adapters used only by Argo `when` expressions and wait targets.
The GitOps release dispatcher consumes the complete JSON contract from the shared workflow volume
and is responsible for translating it into a release intent. Consumers must reject unknown
`contract_version` values.

Ownership stops at this boundary:

- This repo computes offline metrics, logs the MLflow run, registers an immutable model version,
  compares it with the current champion, and emits both candidate and baseline model versions. The
  baseline is deployment-neutral provenance that lets GitOps restore the last accepted model on
  rollback; a bootstrap result has `baseline_model_version: null`.
- GitOps release automation validates the contract, generates smoke traffic, evaluates the online
  candidate, promotes the registry alias, and renders the desired deployment state.
- This repo never receives GitHub App credentials and never knows a GitOps manifest path.
