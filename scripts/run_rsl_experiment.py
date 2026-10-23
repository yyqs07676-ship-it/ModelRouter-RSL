"""Cold-start vs learned-policy experiment for the RSL router.

Protocol
--------
1. reset the learned policy (all Q = 0.5)
2. cold evaluation on a held-out set with exploration disabled
3. learning phase on a disjoint training set (epsilon-greedy enabled)
4. warm evaluation on the same held-out set, again with exploration disabled

The score comes from the deterministic verifier (math), so the comparison is
not affected by LLM-judge noise.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
BASE = "http://127.0.0.1:8000"
HEADERS = {"Authorization": "Bearer dev-router-key"}

TRAINING = [
    # gcd
    "计算 48 和 18 的最大公约数，并说明步骤。",
    "计算 96 和 36 的最大公约数，并说明步骤。",
    "计算 54 和 24 的最大公约数，并说明步骤。",
    "计算 84 和 30 的最大公约数，并说明步骤。",
    "计算 72 和 48 的最大公约数，并说明步骤。",
    "计算 120 和 45 的最大公约数，并说明步骤。",
    "计算 81 和 27 的最大公约数，并说明步骤。",
    "计算 100 和 75 的最大公约数，并说明步骤。",
    "计算 36 和 24 的最大公约数，并说明步骤。",
    "计算 144 和 60 的最大公约数，并说明步骤。",
    # addition
    "计算 37 + 58 等于多少？",
    "计算 63 + 29 等于多少？",
    "计算 145 + 87 等于多少？",
    "计算 208 + 97 等于多少？",
    "计算 99 + 76 等于多少？",
    "计算 134 + 58 等于多少？",
    "计算 77 + 45 等于多少？",
    "计算 156 + 39 等于多少？",
    # multiplication
    "计算 12 × 13 等于多少？",
    "计算 17 × 6 等于多少？",
    "计算 24 × 15 等于多少？",
    "计算 19 × 23 等于多少？",
    "计算 14 × 16 等于多少？",
    "计算 21 × 18 等于多少？",
]

# Held out from the training set.
EVALUATION = [
    "计算 56 和 42 的最大公约数，并说明步骤。",
    "计算 90 和 36 的最大公约数，并说明步骤。",
    "计算 187 + 64 等于多少？",
    "计算 59 + 88 等于多少？",
    "计算 27 × 16 等于多少？",
    "计算 15 × 14 等于多少？",
]


def post(path: str, payload: dict, params: dict | None = None) -> dict:
    response = httpx.post(
        f"{BASE}{path}",
        headers=HEADERS,
        json=payload,
        params=params,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def ask(prompt: str, explore: bool, learn: bool) -> dict:
    data = post(
        "/v1/rsl/chat",
        {"model": "rsl-auto", "messages": [{"role": "user", "content": prompt}]},
        params={
            "explore": "true" if explore else "false",
            "learn": "true" if learn else "false",
        },
    )
    rsl = data.get("rsl") or {}
    return {
        "prompt": prompt,
        "model": rsl.get("selected_model"),
        "task_type": rsl.get("task_type"),
        "score": rsl.get("final_score"),
        "mode": rsl.get("selection_mode"),
        "q_before": rsl.get("q_before"),
        "q_after": rsl.get("q_after"),
        "answer_tail": (data["choices"][0]["message"]["content"].strip().splitlines() or [""])[-1][:120],
    }


def evaluate(label: str) -> list[dict]:
    rows = []
    for index, prompt in enumerate(EVALUATION, 1):
        row = ask(prompt, explore=False, learn=False)
        rows.append(row)
        print(f"  [{label} {index}/{len(EVALUATION)}] {row['model']} score={row['score']}")
    return rows


def summarize(rows: list[dict]) -> dict:
    scores = [row["score"] for row in rows if row["score"] is not None]
    return {
        "avg_score": round(statistics.mean(scores), 2) if scores else None,
        "models": {model: sum(1 for row in rows if row["model"] == model) for model in {row["model"] for row in rows}},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("== reset ==")
    post("/v1/rsl/reset", {})

    print("== cold evaluation (exploration disabled) ==")
    cold = evaluate("cold")

    # Shuffle once with a fixed seed: otherwise the hard GCD block comes first
    # and the easy arithmetic block is always sampled by whichever arm happens
    # to be exploited late, which biases the Q-table.
    training = list(TRAINING)
    random.Random(2026).shuffle(training)
    epochs = 2
    training = training * epochs

    print(f"== learning (epsilon-greedy, {len(training)} requests, {epochs} epochs) ==")
    learning = []
    for index, prompt in enumerate(training, 1):
        row = ask(prompt, explore=True, learn=True)
        learning.append(row)
        print(f"  [{index}/{len(training)}] {row['model']} ({row['mode']}) score={row['score']} Q {row['q_before']}->{row['q_after']}")

    print("== warm evaluation (exploration disabled) ==")
    warm = evaluate("warm")

    stats = httpx.get(f"{BASE}/v1/rsl/stats", headers=HEADERS, timeout=30).json()

    result = {
        "config": stats.get("config"),
        "cold": cold,
        "learning": learning,
        "warm": warm,
        "cold_summary": summarize(cold),
        "warm_summary": summarize(warm),
        "policy": stats.get("policy"),
    }
    (OUT / "rsl_experiment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# RSL 路由学习实验",
        "",
        "## 结果对比",
        "",
        "| 阶段 | 平均得分 | 模型选择 |",
        "| --- | --- | --- |",
        f"| 冷启动 | {result['cold_summary']['avg_score']} | {result['cold_summary']['models']} |",
        f"| 学习后 | {result['warm_summary']['avg_score']} | {result['warm_summary']['models']} |",
        "",
        "## 最终 Q 表",
        "",
        "| 任务类型 | 模型 | Q 值 | 样本数 | 平均质量 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in stats.get("policy", []):
        lines.append(
            f"| {row['task_type']} | {row['routing_key']} | {row['q_value']:.4f} | "
            f"{row['sample_count']} | {row['avg_quality']:.3f} |"
        )
    (OUT / "rsl_experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n== summary ==")
    print("cold:", result["cold_summary"])
    print("warm:", result["warm_summary"])
    print("saved:", OUT / "rsl_experiment.json")
    print("saved:", OUT / "rsl_experiment.md")


if __name__ == "__main__":
    sys.exit(main())
