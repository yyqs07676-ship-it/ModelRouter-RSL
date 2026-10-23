"""Unit tests for the RSL engine, verifier and judge parsing."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.rsl import RSLEngine, classify_task, parse_judge_output  # noqa: E402
from app.core.rsl_verifier import verify_answer  # noqa: E402


def test_classify_task() -> None:
    assert classify_task("请用 Python 写一个排序函数") == "coding"
    assert classify_task("把这句话翻译成英文") == "translation"
    assert classify_task("计算 48 和 18 的最大公约数") == "reasoning"
    assert classify_task("写一篇关于杭州的短文") == "writing"
    assert classify_task("你好") == "general"


def test_parse_judge_json() -> None:
    score, reason = parse_judge_output('{"score": 8, "reason": "回答准确"}')
    assert score == 8.0
    assert reason == "回答准确"


def test_parse_judge_chinese_text() -> None:
    score, _ = parse_judge_output("我认为这个答案评分：7 分")
    assert score == 7.0


def test_parse_judge_garbage_returns_none() -> None:
    score, _ = parse_judge_output("完全无法解析的输出")
    assert score is None


def test_verifier_gcd_correct() -> None:
    verdict = verify_answer("计算 48 和 18 的最大公约数", "所以，48 和 18 的最大公约数是 6。")
    assert verdict is not None
    assert verdict[0] == 10.0


def test_verifier_gcd_wrong() -> None:
    verdict = verify_answer("计算 48 和 18 的最大公约数", "所以最大公约数是 18。")
    assert verdict is not None
    assert verdict[0] == 2.0


def test_verifier_addition_uses_last_conclusion() -> None:
    # Regression: "63 加 29 等于 92" once extracted 63 as the conclusion.
    verdict = verify_answer("计算 63 + 29 等于多少？", "63 加 29 等于 92。")
    assert verdict is not None
    assert verdict[0] == 10.0


def test_verifier_multiplication_wrong() -> None:
    verdict = verify_answer("计算 12 × 13 等于多少？", "12 乘 13 的结果是 25。")
    assert verdict is not None
    assert verdict[0] == 2.0


def test_verifier_returns_none_for_open_ended() -> None:
    assert verify_answer("写一篇关于杭州的短文", "杭州很美。") is None


def test_exploit_picks_best_q_value() -> None:
    engine = RSLEngine(None, None, candidates=["small", "strong"], epsilon=0.0, seed=1)

    async def fake_q(_task: str):
        return {
            "small": {"q_value": 0.3, "sample_count": 5},
            "strong": {"q_value": 0.8, "sample_count": 2},
        }

    engine.get_q_values = fake_q  # type: ignore[assignment]
    model, mode = asyncio.run(engine.select_model("reasoning"))
    assert model == "strong"
    assert mode == "exploit"


def test_tie_break_prefers_less_sampled_arm() -> None:
    engine = RSLEngine(None, None, candidates=["small", "strong"], epsilon=0.0, seed=1)

    async def fake_q(_task: str):
        return {
            "small": {"q_value": 0.5, "sample_count": 9},
            "strong": {"q_value": 0.5, "sample_count": 0},
        }

    engine.get_q_values = fake_q  # type: ignore[assignment]
    model, mode = asyncio.run(engine.select_model("reasoning"))
    assert model == "strong"
    assert mode == "exploit"
