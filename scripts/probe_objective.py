"""Probe tasks that can be checked without an LLM judge."""

from __future__ import annotations

import json
import re

import httpx

BASE = "http://127.0.0.1:8000"
HEADERS = {"Authorization": "Bearer dev-router-key"}

TRANSLATIONS = [
    ("把这句话翻译成中文：The cat is sleeping on the sofa.", ["猫", "沙发"]),
    ("把这句话翻译成中文：I will go to the library tomorrow.", ["明天", "图书馆"]),
    ("把这句话翻译成英文：今天天气很好。", ["weather", "nice"]),
    ("把这句话翻译成英文：我们明天开会。", ["meeting", "tomorrow"]),
]

JSON_TASKS = [
    ("从这句话提取人物、地点、时间，只输出 JSON，字段为 person/location/time：张三明天下午三点在北京开会。",
     ["person", "location", "time"]),
    ("从这句话提取人物、地点、时间，只输出 JSON，字段为 person/location/time：李四周五在上海出差。",
     ["person", "location", "time"]),
    ("请只输出 JSON，包含字段 name 和 age：小明今年 18 岁。", ["name", "age"]),
    ("请只输出 JSON，包含字段 city 和 population：杭州人口约 1200 万。", ["city", "population"]),
]

CODING = [
    ("用 Python 写一个函数 is_prime(n)，判断素数，只输出代码。", ["def", "return"]),
    ("用 Python 写一个函数 add(a, b)，返回两数之和，只输出代码。", ["def", "return"]),
]


def ask(model: str, prompt: str) -> str:
    response = httpx.post(
        f"{BASE}/v1/chat/completions",
        headers=HEADERS,
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def score_keywords(answer: str, keywords: list[str], lowercase: bool = False) -> float:
    text = answer.lower() if lowercase else answer
    hits = sum(1 for keyword in keywords if (keyword.lower() if lowercase else keyword) in text)
    if hits == len(keywords):
        return 10.0
    if hits > 0:
        return 5.0
    return 2.0


def score_json(answer: str, fields: list[str]) -> float:
    match = re.search(r"\{[\s\S]*\}", answer)
    if not match:
        return 2.0
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 2.0
    return 10.0 if all(field in data for field in fields) else 5.0


def main() -> None:
    buckets = {"translation": [], "json": [], "coding": []}
    for model in ("qwen-small", "qwen-strong"):
        scores = {"translation": [], "json": [], "coding": []}
        for prompt, keywords in TRANSLATIONS:
            text = ask(model, prompt)
            scores["translation"].append(score_keywords(text, keywords, lowercase=True))
        for prompt, fields in JSON_TASKS:
            text = ask(model, prompt)
            scores["json"].append(score_json(text, fields))
        for prompt, keywords in CODING:
            text = ask(model, prompt)
            scores["coding"].append(score_keywords(text, keywords))
        for key, values in scores.items():
            avg = round(sum(values) / len(values), 2)
            buckets[key].append((model, avg))
            print(model, key, avg)
    print(json.dumps(buckets, ensure_ascii=False))


if __name__ == "__main__":
    main()
