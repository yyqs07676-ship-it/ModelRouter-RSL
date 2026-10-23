"""Deterministic answer verifiers used as a trustworthy reward source.

An LLM judge is useful for open-ended answers but unreliable for arithmetic.
When a question has a machine-checkable form, we compute the ground truth with
Python and score against it. Otherwise the caller falls back to the LLM judge.
"""

from __future__ import annotations

import math
import re


def _all_ints(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"-?\d+", text or "")]


def _conclusion_int(answer: str) -> int | None:
    """Extract the number the answer actually concludes with."""
    text = answer or ""

    # 1) Explicit GCD conclusion wins over other numbers in the derivation.
    gcd_matches = re.findall(r"最大公约数(?:是|为|等于|=)\s*(-?\d+)", text)
    if gcd_matches:
        return int(gcd_matches[-1])

    # 2) "等于 X" / "= X": the LAST occurrence is the conclusion, because the
    #    derivation may contain earlier "63 + 29 等于 ..." style fragments.
    equals = re.findall(r"(?:等于|=)\s*(-?\d+)", text)
    if equals:
        return int(equals[-1])

    # 3) "结果是 X" / "答案是 X"
    results = re.findall(r"(?:答案|结果)(?:是|为|等于)\s*(-?\d+)", text)
    if results:
        return int(results[-1])

    # 4) A sentence ending with "是 X"
    trailing = re.findall(r"(?:是|为)\s*(-?\d+)\s*[。.!！]?\s*$", text.strip())
    if trailing:
        return int(trailing[-1])

    numbers = _all_ints(answer)
    return numbers[-1] if numbers else None


def _score(correct: bool, expected: int, got: int | None, label: str) -> tuple[float, str]:
    if correct:
        return 10.0, f"规则校验通过：{label} = {expected}"
    return 2.0, f"规则校验未通过：{label} 应为 {expected}，答案给出 {got}"


def verify_answer(question: str, answer: str) -> tuple[float, str] | None:
    """Return (score, reason) when the question is machine-checkable."""
    text = question or ""

    if ("最大公约数" in text or re.search(r"\bGCD\b", text, re.IGNORECASE)):
        numbers = _all_ints(text)
        if len(numbers) >= 2:
            expected = math.gcd(abs(numbers[0]), abs(numbers[1]))
            got = _conclusion_int(answer)
            return _score(got == expected, expected, got, "最大公约数")

    addition = re.search(r"(\d+)\s*[\+加]\s*(\d+)", text)
    if addition:
        expected = int(addition.group(1)) + int(addition.group(2))
        got = _conclusion_int(answer)
        return _score(got == expected, expected, got, "加法结果")

    multiplication = re.search(r"(\d+)\s*[×xX*乘]\s*(\d+)", text)
    if multiplication:
        expected = int(multiplication.group(1)) * int(multiplication.group(2))
        got = _conclusion_int(answer)
        return _score(got == expected, expected, got, "乘法结果")

    return None
