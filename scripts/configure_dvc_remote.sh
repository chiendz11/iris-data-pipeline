#!/usr/bin/env bash
set -euo pipefail

: "${DVC_BUCKET:?Set DVC_BUCKET to the Terraform dvc_bucket output}"
dvc remote modify aws url "s3://${DVC_BUCKET}/dvc-cache"
echo "DVC remote now uses s3://${DVC_BUCKET}/dvc-cache; AWS credentials come from your profile or IRSA."
