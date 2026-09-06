from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import unquote_plus

import boto3


def fetch(bucket: str, key: str, destination: str | Path) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, unquote_plus(key), str(target))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the S3 dataset that triggered Argo.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--destination", default="data/iris.csv")
    args = parser.parse_args()
    print(fetch(args.bucket, args.key, args.destination))


if __name__ == "__main__":
    main()
