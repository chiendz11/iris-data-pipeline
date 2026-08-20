# Iris data and training pipeline

Repository 1/3 của capstone MLOps. Repo này sở hữu **dữ liệu, feature preparation,
training, evaluation và quality gate**; không sở hữu MLflow server hay inference API.

## Vì sao dùng Iris + Logistic Regression?

- Iris có 150 bản ghi, 4 feature số, 3 lớp cân bằng và nằm sẵn trong scikit-learn.
- CI không phụ thuộc Kaggle/API key/network và train chỉ mất vài giây.
- Dataset không chứa PII, phù hợp để public repo và demo DVC.
- Logistic Regression dễ giải thích, model nhỏ, hỗ trợ multiclass và đạt accuracy ổn định > 0.90.
- Bài toán vẫn đủ thật để demo version dữ liệu, experiment, metric gate, model registry và serving.

Không chọn deep learning vì thời gian build image, tải weight và nhu cầu CPU/GPU sẽ làm CI chậm
mà không bổ sung giá trị MLOps tương ứng cho capstone đầu tiên.

## Chạy nhanh

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
REGISTER_MODEL=false MLFLOW_TRACKING_URI=file:./mlruns dvc repro
cat metrics.json
```

Để log và đăng ký model vào repo `model-registry` đang chạy:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export REGISTER_MODEL=true
python -m src.prepare
python -m src.train
```

Model đạt `quality_gate.min_accuracy` sẽ được gắn alias `champion`. Mọi model mới đều
được gắn alias `candidate` để có thể kiểm tra trước khi promote.

## DVC remote

`.dvc/config` trỏ tới bucket `dvc` trên MinIO local. Credential không commit vào Git:

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
dvc push
```

Trong môi trường thật, thay credential mặc định và đưa chúng vào secret manager.

## CI/CD

GitHub Actions chạy lint, unit test, `dvc repro` offline, build image và khi merge `main`
sẽ push image training lên GHCR. Argo `argo/iris-training-workflow.yaml` dùng image đó để
chạy prepare → train/evaluate → verify trên Kubernetes.

