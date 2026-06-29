#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark async NIXL KV pulls through the Dynamo frontend.

Runs inside the Dynamo container against an already-started disaggregated stack.
It reads /tmp/phase3.env unless the same values are supplied in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


REQUIRED_VARS = (
    "PHASE3_FRONTEND_URL",
    "PHASE3_MODEL_NAME",
    "PHASE3_DECODE_LOG",
)

PULL_SUBMIT_RE = re.compile(
    r"DISAGG_DECODE_PULL_SUBMIT\s+request_id=(\d+)\s+prompt_tokens=(\d+)\s+blocks=(\d+)\s+pending_imports=(\d+)"
)
IMPORT_RE = re.compile(
    r"DISAGG_DECODE_IMPORT\s+request_id=(\d+)\s+prompt_tokens=(\d+)\s+imported_blocks=(\d+)\s+hashes_registered=(\d+)"
)


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = Path(os.environ.get("PHASE3_ENV_FILE", "/tmp/phase3.env"))
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip('"').strip("'")
    for key in REQUIRED_VARS:
        if key in os.environ:
            env[key] = os.environ[key]
    missing = [key for key in REQUIRED_VARS if key not in env]
    if missing:
        raise RuntimeError(f"Missing required environment values: {missing}")
    return env


def _parse_bursts(value: str) -> list[int]:
    bursts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not bursts or any(count <= 0 for count in bursts):
        raise ValueError("--bursts must contain positive integers")
    return bursts


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _build_prompt(index: int, prompt_words: int) -> str:
    token = f"async-pull-benchmark-{index}"
    body = " ".join([token] * prompt_words)
    return f"{body}\nReturn one concise sentence."


def _stream_request(
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    index: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    first_chunk = None
    chunks = 0
    output_chars = 0
    finish_reason = None
    with requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            payload_text = raw[len("data:") :].strip()
            if payload_text == "[DONE]":
                break
            if first_chunk is None:
                first_chunk = time.perf_counter()
            chunks += 1
            chunk = json.loads(payload_text)
            choice = chunk.get("choices", [{}])[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            output_chars += len(choice.get("delta", {}).get("content") or "")
    finished = time.perf_counter()
    if chunks == 0:
        raise RuntimeError(f"request {index} produced no stream chunks")
    return {
        "index": index,
        "ttft_s": (first_chunk - started) if first_chunk is not None else None,
        "latency_s": finished - started,
        "chunks": chunks,
        "output_chars": output_chars,
        "finish_reason": finish_reason,
    }


def _read_decode_markers(path: str) -> dict[str, Any]:
    try:
        text = Path(path).read_text()
    except OSError:
        text = ""
    submits = PULL_SUBMIT_RE.findall(text)
    imports = IMPORT_RE.findall(text)
    pending_depths = [int(row[3]) for row in submits]
    return {
        "pull_submit_count": len(submits),
        "import_count": len(imports),
        "max_pending_imports": max(pending_depths) if pending_depths else 0,
        "submitted_blocks": sum(int(row[2]) for row in submits),
        "imported_blocks": sum(int(row[2]) for row in imports),
        "registered_hashes": sum(int(row[3]) for row in imports),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bursts", default=os.environ.get("PHASE3_BENCH_BURSTS", "2,4,3"))
    parser.add_argument(
        "--burst-gaps",
        default=os.environ.get("PHASE3_BENCH_BURST_GAPS", "1,3"),
        help="Comma-separated seconds between bursts.",
    )
    parser.add_argument(
        "--prompt-words",
        type=int,
        default=int(os.environ.get("PHASE3_BENCH_PROMPT_WORDS", "240")),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("PHASE3_BENCH_MAX_TOKENS", "16")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("PHASE3_BENCH_TIMEOUT", "300")),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(os.environ.get("PHASE3_BENCH_WARMUP", "1")),
    )
    parser.add_argument("--output", default=os.environ.get("PHASE3_BENCH_OUTPUT", ""))
    args = parser.parse_args()

    env = _load_env()
    bursts = _parse_bursts(args.bursts)
    gaps = [float(part.strip()) for part in args.burst_gaps.split(",") if part.strip()]
    if len(gaps) < max(0, len(bursts) - 1):
        gaps.extend([0.0] * (len(bursts) - 1 - len(gaps)))

    url = env["PHASE3_FRONTEND_URL"]
    model = env["PHASE3_MODEL_NAME"]
    decode_log = env["PHASE3_DECODE_LOG"]

    requests.get(f"{url}/v1/models", timeout=30).raise_for_status()

    for warmup_idx in range(args.warmup):
        _stream_request(
            url=url,
            model=model,
            prompt=_build_prompt(-(warmup_idx + 1), args.prompt_words),
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            index=-(warmup_idx + 1),
        )

    total_requests = sum(bursts)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(bursts)) as pool:
        futures = []
        request_index = 0
        for burst_idx, count in enumerate(bursts):
            print(f"[bench] submitting burst {burst_idx + 1}/{len(bursts)}: {count} requests")
            for _ in range(count):
                futures.append(
                    pool.submit(
                        _stream_request,
                        url=url,
                        model=model,
                        prompt=_build_prompt(request_index, args.prompt_words),
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        index=request_index,
                    )
                )
                request_index += 1
            if burst_idx < len(bursts) - 1 and gaps[burst_idx] > 0:
                time.sleep(gaps[burst_idx])

        for future in as_completed(futures, timeout=args.timeout * max(1, total_requests)):
            result = future.result()
            results.append(result)
            print(
                "[bench] request {index} done latency={latency_s:.3f}s ttft={ttft_s:.3f}s chunks={chunks}".format(
                    **result
                )
            )

    elapsed = time.perf_counter() - started
    latencies = [float(row["latency_s"]) for row in results]
    ttfts = [float(row["ttft_s"]) for row in results if row["ttft_s"] is not None]
    markers = _read_decode_markers(decode_log)
    summary = {
        "requests": len(results),
        "bursts": bursts,
        "burst_gaps_s": gaps[: max(0, len(bursts) - 1)],
        "prompt_words": args.prompt_words,
        "max_tokens": args.max_tokens,
        "elapsed_s": elapsed,
        "request_rate": len(results) / elapsed if elapsed > 0 else 0.0,
        "latency_s": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 50),
            "p90": _percentile(latencies, 90),
            "p99": _percentile(latencies, 99),
            "max": max(latencies) if latencies else 0.0,
        },
        "ttft_s": {
            "avg": statistics.mean(ttfts) if ttfts else 0.0,
            "p50": _percentile(ttfts, 50),
            "p90": _percentile(ttfts, 90),
            "p99": _percentile(ttfts, 99),
            "max": max(ttfts) if ttfts else 0.0,
        },
        "decode_log": markers,
    }

    print("[bench] summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if markers["pull_submit_count"] < total_requests:
        print("[bench] ERROR: fewer pull submissions than benchmark requests", file=sys.stderr)
        return 2
    if markers["import_count"] < total_requests:
        print("[bench] ERROR: fewer completed imports than benchmark requests", file=sys.stderr)
        return 3
    if markers["max_pending_imports"] < 2 and total_requests > 1:
        print("[bench] ERROR: async imports did not overlap", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
