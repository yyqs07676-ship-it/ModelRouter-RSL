"""Quick A/B probe: does the strong model actually beat the small one?"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.rsl_verifier import verify_answer  # noqa: E402

BASE = "http://127.0.0.1:8000"
HEADERS = {"Authorization": "Bearer dev-router-key"}

PROMPTS = [
    "计算 48 和 18 的最大公约数，并说明步骤。",
    "计算 96 和 36 的最大公约数，并说明步骤。",
    "计算 54 和 24 的最大公约数，并说明步骤。",
    "计算 84 和 30 的最大公约数，并说明步骤。",
    "计算 37 + 58 等于多少？",
    "计算 63 + 29 等于多少？",
    "计算 145 + 87 等于多少？",
    "计算 208 + 97 等于多少？",
    "计算 12 × 13 等于多少？",
    "计算 17 × 6 等于多少？",
    "计算 24 × 15 等于多少？",
    "计算 19 × 23 等于多少？",
]


def ask(model: str, prompt: str) -> str:
    response = httpx.post(
        f"{BASE}/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main() -> None:
    rows = []
    for prompt in PROMPTS:
        row = {"prompt": prompt}
        for model in ("qwen-small", "qwen-strong"):
            answer = ask(model, prompt)
            verdict = verify_answer(prompt, answer)
            row[model] = {
                "score": verdict[0] if verdict else None,
                "reason": verdict[1] if verdict else "",
                "tail": answer.strip().splitlines()[-1][:80],
            }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    small = [r["qwen-small"]["score"] for r in rows if r["qwen-small"]["score"] is not None]
    strong = [r["qwen-strong"]["score"] for r in rows if r["qwen-strong"]["score"] is not None]
    print("\nqwen-small 平均分:", round(sum(small) / len(small), 2) if small else None)
    print("qwen-strong 平均分:", round(sum(strong) / len(strong), 2) if strong else None)


if __name__ == "__main__":
    main()
