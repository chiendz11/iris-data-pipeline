from __future__ import annotations

import argparse
import json

import requests


def query(prometheus_url: str, expression: str) -> float:
    response = requests.get(
        f"{prometheus_url.rstrip('/')}/api/v1/query",
        params={"query": expression},
        timeout=15,
    )
    response.raise_for_status()
    result = response.json()["data"]["result"]
    return float(result[0]["value"][1]) if result else 0.0


def evaluate(prometheus_url: str, service: str, max_p95: float, max_error_rate: float) -> dict:
    selector = f'service="{service}"'
    rps = query(prometheus_url, f'sum(rate(iris_prediction_requests_total{{{selector}}}[5m]))')
    errors = query(
        prometheus_url,
        f'sum(rate(iris_prediction_requests_total{{{selector},status!="success"}}[5m]))',
    )
    p95 = query(
        prometheus_url,
        "histogram_quantile(0.95, "
        f"sum by (le) (rate(iris_prediction_latency_seconds_bucket{{{selector}}}[5m])))",
    )
    error_rate = errors / rps if rps > 0 else 0.0
    passed = rps > 0 and p95 <= max_p95 and error_rate <= max_error_rate
    return {"passed": passed, "rps": rps, "p95_seconds": p95, "error_rate": error_rate}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prometheus-url", required=True)
    parser.add_argument("--service", default="iris-classifier")
    parser.add_argument("--max-p95", type=float, default=0.5)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--output", default="/tmp/canary-result.json")
    parser.add_argument("--passed-output", default="/tmp/canary-passed.txt")
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.prometheus_url, args.service, args.max_p95, args.max_error_rate)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream)
    with open(args.passed_output, "w", encoding="utf-8") as stream:
        stream.write(str(result["passed"]).lower())
    print(json.dumps(result))
    if args.fail_on_reject and not result["passed"]:
        raise SystemExit("Canary SLO gate failed")


if __name__ == "__main__":
    main()
