from __future__ import annotations

import argparse
import random
import time

import requests

SAMPLES = [
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.7, 3.1, 5.6, 2.4],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()
    for _ in range(args.requests):
        response = requests.post(
            args.url,
            json={"instances": [random.choice(SAMPLES)]},  # nosec: load generation only
            timeout=5,
        )
        response.raise_for_status()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
