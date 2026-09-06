# Iris data and training pipeline (repo 1/5)

Repository 1/5 của capstone MLOps. Repo này sở hữu **dữ liệu, feature preparation,
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

Model mới luôn được đăng ký và gắn alias `candidate`. Nếu accuracy đạt ngưỡng tuyệt đối và
cao hơn `champion` ít nhất `min_improvement`, model chỉ được gắn tag
`quality_gate=passed`; bước train **không tự đổi champion**. Argo chỉ đổi alias `champion`
sau khi canary online cũng qua ngưỡng latency/error rate.

## DVC + S3 và trigger dataset

`.dvc/config` có remote AWS mẫu. Lấy output `dvc_bucket` từ Terraform rồi cấu hình:

```bash
export DVC_BUCKET=iris-mlops-dev-dvc-xxxxxxxx
./scripts/configure_dvc_remote.sh
dvc push
```

CI dùng GitHub OIDC để nhận credential ngắn hạn. Nó push DVC cache vào `dvc-cache/`, rồi
copy dataset đã materialize vào `datasets/iris.csv`. Chỉ prefix thứ hai phát S3 notification,
tránh mỗi object hash của DVC tạo một training run.

## CI/CD

GitHub Actions chạy lint/test/`dvc repro`, push image bất biến lên ECR, push dữ liệu lên S3.
S3 gửi event vào SQS; Argo Events Sensor tạo production workflow được quản lý tại
`iris-gitops/environments/production/data-pipeline`:

`fetch S3 → train/log MLflow → offline gate → GitOps PR 10% → Argo CD/KServe → smoke traffic
→ Prometheus gate → promote MLflow → GitOps PR 100% → Argo CD/KServe`.

Training image chỉ xuất contract file `model-version.txt`, `quality-gate.txt` và
`bootstrap-required.txt`; nó không xác thực GitHub App, gửi event, biết đường dẫn/cấu trúc
InferenceService hay mở GitOps PR. Dispatcher image cùng WorkflowTemplate do `iris-gitops` sở hữu
nhận các output này và phát release intent `{action, model_version, change_id}`. GitOps CI cùng
CODEOWNER phải duyệt, reviewer merge, rồi Argo CD reconcile các field deployment được quản lý trong
Git. Argo Workflow chỉ có Kubernetes RBAC `get/list/watch` để xác nhận desired state đã được
reconcile và KServe đã Ready, không còn `kubectl patch`.

Nếu online gate thất bại, workflow mở GitOps PR đặt canary traffic về 0 và alias champion vẫn trỏ
tới model cũ.
Riêng model đầu tiên chưa có stable revision để chia 90/10 nên dùng nhánh bootstrap 100%; mọi
model tiếp theo mới đi qua canary 10% bắt buộc.
Pod dùng IRSA `iris-training` để đọc S3; không có access key trong manifest.

GitHub environment `prod` variables: `AWS_REGION`, `AWS_DEPLOY_ROLE_ARN`,
`TRAINING_ECR_REPOSITORY`, `DVC_BUCKET`.

Repo này không giữ manifest production và không có quyền mutate Kubernetes. S3 key chứa commit SHA;
Sensor trong `iris-gitops` ghép SHA đó với ECR repository để mỗi dataset luôn chạy đúng training
image đã build từ cùng commit.

Model promotion dùng GitHub App riêng chỉ có `Contents/Pull requests: write` trên `iris-gitops`.
Private key nằm trong AWS Secrets Manager và chỉ được inject vào Dispatch Pod chạy image riêng do
DevOps build; fetch/train/smoke/evaluate containers không nhận credential này. GitOps workflow đọc
cùng secret bằng OIDC role read-only để mint token ngắn hạn. Key không nằm trong training image,
repo, Terraform state hay GitHub Actions variables.
