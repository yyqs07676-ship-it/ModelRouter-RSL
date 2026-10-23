"""Recursive Self-Learning (RSL) engine for the Model Router.

Two nested learning loops:

1. Intra-request recursion: generate -> judge -> revise -> judge, until the
   answer is good enough or the round budget is exhausted.
2. Cross-request learning: every finished request updates a Q-table keyed by
   (task_type, routing_key), so later requests route to the model that has
   historically handled that task type better.

The model weights are never changed; this is policy/experience learning.
"""

from __future__ import annotations

import json
import random
import re
import time
import uuid
from datetime import datetime, timezone

from app.schemas.chat import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Usage,
)
from app.core.rsl_verifier import verify_answer


TASK_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "代码", "编程", "函数", "算法", "报错", "bug", "debug", "python",
        "java", "sql", "接口", "脚本", "编译", "重构",
    ],
    "translation": [
        "翻译", "中译英", "英译中", "translate", "translation", "英文", "日文",
    ],
    "reasoning": [
        "计算", "数学", "证明", "推理", "逻辑", "解方程", "概率", "多少",
        "为什么", "分析一下", "推导",
    ],
    "writing": [
        "写一篇", "作文", "文案", "故事", "诗", "润色", "改写", "摘要", "总结",
    ],
}

JUDGE_SYSTEM = """你是一个严格的中文答案评审员。
请先独立核对答案的正确性，再按准确性、完整性、是否直接回答问题评分：
- 存在事实错误或计算错误的，score 不得高于 3；
- 答非所问或明显不完整的，score 不得高于 5；
- 只有正确、完整、直接回答问题的答案，才可以给 7 分以上。
只输出一行 JSON，不要输出任何其他文字：
{"score": 0到10之间的数字, "reason": "简短中文理由"}
"""

JUDGE_NUMERIC_FALLBACK = """你是一个严格的答案评审员。
请只输出一个 0 到 10 之间的数字，表示下面这个回答的正确性与完整性，不要输出任何其他字符。
"""

REVISE_SYSTEM = """你正在根据评审意见修订答案。
请直接输出修订后的最终答案，不要解释修改过程，不要输出 JSON。"""


def classify_task(text: str) -> str:
    """Keyword task classifier. Deterministic and easy to explain/graded."""
    lowered = (text or "").lower()
    best_category = "general"
    best_hits = 0
    for category, keywords in TASK_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
        if hits > best_hits:
            best_hits = hits
            best_category = category
    return best_category


def parse_judge_output(text: str) -> tuple[float | None, str]:
    """Parse a judge reply into (score 0..10, reason).

    Small local models do not always return strict JSON, so this falls back to
    regular expressions. Returns (0.0, raw_text) when nothing can be parsed.
    """
    raw = (text or "").strip()
    if not raw:
        return None, ""

    candidate = raw
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidate = raw[start:end + 1]

    try:
        data = json.loads(candidate)
        score = float(data.get("score"))
        reason = str(data.get("reason", "")).strip()
        return max(0.0, min(10.0, score)), reason
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

    match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw)
    if not match:
        match = re.search(r'(?:分数|评分|score)\D{0,8}(\d+(?:\.\d+)?)', raw, re.IGNORECASE)
    if not match:
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:/\s*10|分)', raw)
    if match:
        return max(0.0, min(10.0, float(match.group(1)))), raw[:300]

    return None, raw[:300]


class RSLEngine:
    def __init__(
        self,
        db,
        router_engine,
        *,
        candidates: list[str] | None = None,
        judge_model: str = "qwen-strong",
        alpha: float = 0.25,
        epsilon: float = 0.25,
        threshold: float = 0.75,
        max_rounds: int = 2,
        latency_penalty: float = 0.1,
        temperature: float = 0.2,
        seed: int | None = None,
    ):
        self.db = db
        self.router = router_engine
        self.candidates = candidates or ["qwen-small", "qwen-strong"]
        self.judge_model = judge_model
        self.alpha = alpha
        self.epsilon = epsilon
        self.threshold = threshold
        self.max_rounds = max(1, max_rounds)
        self.latency_penalty = latency_penalty
        self.temperature = temperature
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ setup
    async def _commit(self) -> None:
        conn = getattr(self.db, "_conn", None)
        if conn is not None:
            await conn.commit()

    async def initialize(self) -> None:
        await self.db._execute(
            """
            CREATE TABLE IF NOT EXISTS rsl_policy (
                task_type TEXT NOT NULL,
                routing_key TEXT NOT NULL,
                q_value REAL NOT NULL DEFAULT 0.5,
                sample_count INTEGER NOT NULL DEFAULT 0,
                total_quality REAL NOT NULL DEFAULT 0,
                total_latency_ms REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (task_type, routing_key)
            )
            """
        )
        await self.db._execute(
            """
            CREATE TABLE IF NOT EXISTS rsl_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                task_type TEXT NOT NULL,
                routing_key TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                quality REAL NOT NULL,
                reward REAL NOT NULL,
                latency_ms REAL NOT NULL,
                q_before REAL,
                q_after REAL,
                judge_reason TEXT
            )
            """
        )
        for task_type in list(TASK_KEYWORDS.keys()) + ["general"]:
            for routing_key in self.candidates:
                await self.db._execute(
                    """
                    INSERT OR IGNORE INTO rsl_policy (task_type, routing_key)
                    VALUES (?, ?)
                    """,
                    (task_type, routing_key),
                )
        await self._commit()

    # --------------------------------------------------------------- policy
    async def _policy_row(self, task_type: str, routing_key: str):
        return await self.db._fetchone(
            "SELECT q_value, sample_count FROM rsl_policy WHERE task_type=? AND routing_key=?",
            (task_type, routing_key),
        )

    async def get_q_values(self, task_type: str) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for routing_key in self.candidates:
            row = await self._policy_row(task_type, routing_key)
            if row:
                result[routing_key] = {
                    "q_value": round(float(row["q_value"]), 4),
                    "sample_count": int(row["sample_count"]),
                }
            else:
                result[routing_key] = {"q_value": 0.5, "sample_count": 0}
        return result

    async def select_model(
        self, task_type: str, *, epsilon_override: float | None = None
    ) -> tuple[str, str]:
        """Return (routing_key, mode) with mode in {"explore", "exploit"}."""
        epsilon = self.epsilon if epsilon_override is None else epsilon_override
        if self._rng.random() < epsilon:
            return self._rng.choice(self.candidates), "explore"

        table = await self.get_q_values(task_type)
        # Among equally good arms prefer the one sampled less (optimism).
        best_key = max(
            self.candidates,
            key=lambda key: (table[key]["q_value"], -table[key]["sample_count"]),
        )
        return best_key, "exploit"

    # ----------------------------------------------------------- model calls
    async def _call(
        self,
        routing_key: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> tuple[str, Usage | None, float]:
        provider, model_config = await self.router.route(routing_key)
        request = ChatRequest(
            model=routing_key,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        started = time.monotonic()
        response = await provider.chat_completions(request, model_config)
        latency_ms = (time.monotonic() - started) * 1000
        content = response.choices[0].message.content if response.choices else ""
        return content, response.usage, latency_ms

    async def _judge(self, question: str, answer: str) -> tuple[float, str, str]:
        # Machine-checkable questions get a deterministic score first.
        verified = verify_answer(question, answer)
        if verified is not None:
            score, reason = verified
            return score, reason, "rule_verifier"

        messages = [
            ChatMessage(role="system", content=JUDGE_SYSTEM),
            ChatMessage(
                role="user",
                content=f"用户问题：\n{question}\n\n待评审回答：\n{answer}",
            ),
        ]
        try:
            raw, _, _ = await self._call(
                self.judge_model, messages, temperature=0.0, max_tokens=200
            )
            score, reason = parse_judge_output(raw)
            if score is not None:
                return score, reason, raw

            # Retry once with a numeric-only prompt: small local models often
            # fail strict JSON but can still emit a number.
            retry_messages = [
                ChatMessage(role="system", content=JUDGE_NUMERIC_FALLBACK),
                ChatMessage(
                    role="user",
                    content=f"用户问题：\n{question}\n\n待评审回答：\n{answer}",
                ),
            ]
            raw2, _, _ = await self._call(
                self.judge_model, retry_messages, temperature=0.0, max_tokens=16
            )
            score2, reason2 = parse_judge_output(raw2)
            if score2 is not None:
                return score2, reason2, raw2
            return 5.0, f"judge_unparsed: {raw[:120]}", raw
        except Exception as exc:  # judge failure must not break the request
            return 5.0, f"judge_error: {exc}", ""

    # -------------------------------------------------------------- learning
    async def _update_policy(
        self,
        task_type: str,
        routing_key: str,
        *,
        quality: float,
        latency_ms: float,
        round_index: int,
        reason: str,
    ) -> tuple[float, float]:
        row = await self._policy_row(task_type, routing_key)
        q_before = float(row["q_value"]) if row else 0.5
        count = int(row["sample_count"]) if row else 0

        latency_component = min(latency_ms / 30_000.0, 1.0)
        reward = max(0.0, min(1.0, quality - self.latency_penalty * latency_component))
        q_after = q_before + self.alpha * (reward - q_before)

        await self.db._execute(
            """
            UPDATE rsl_policy
               SET q_value = ?, sample_count = sample_count + 1,
                   total_quality = total_quality + ?,
                   total_latency_ms = total_latency_ms + ?,
                   updated_at = datetime('now')
             WHERE task_type = ? AND routing_key = ?
            """,
            (q_after, quality, latency_ms, task_type, routing_key),
        )
        await self.db._execute(
            """
            INSERT INTO rsl_events
                (timestamp, task_type, routing_key, round_index, quality, reward,
                 latency_ms, q_before, q_after, judge_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                task_type,
                routing_key,
                round_index,
                quality,
                reward,
                latency_ms,
                q_before,
                q_after,
                reason[:500],
            ),
        )
        await self._commit()
        return q_before, q_after

    # ---------------------------------------------------------------- public
    async def chat(
        self,
        request: ChatRequest,
        *,
        epsilon_override: float | None = None,
        learn: bool = True,
    ) -> ChatResponse:
        user_messages = [m for m in request.messages if m.role == "user"]
        question = user_messages[-1].content if user_messages else ""
        task_type = classify_task(question)
        routing_key, mode = await self.select_model(
            task_type, epsilon_override=epsilon_override
        )

        rounds: list[dict] = []
        best_answer = ""
        best_score = -1.0
        best_reason = ""
        total_prompt = 0
        total_completion = 0
        total_latency = 0.0
        previous_answer = ""
        previous_reason = ""
        used_key = routing_key

        for round_index in range(1, self.max_rounds + 1):
            if round_index == 1:
                messages = list(request.messages)
            else:
                messages = [
                    ChatMessage(role="system", content=REVISE_SYSTEM),
                    ChatMessage(
                        role="user",
                        content=(
                            f"原问题：\n{question}\n\n"
                            f"上一版答案：\n{previous_answer}\n\n"
                            f"评审意见：\n{previous_reason}\n\n"
                            "请针对上述意见改进答案。"
                        ),
                    ),
                ]

            try:
                answer, usage, latency_ms = await self._call(
                    used_key,
                    messages,
                    temperature=(
                        self.temperature
                        if round_index == 1
                        else max(0.0, self.temperature * 0.5)
                    ),
                    max_tokens=request.max_tokens or 512,
                )
            except Exception:
                # One fallback attempt on the other candidate, then give up.
                fallback = next(
                    (key for key in self.candidates if key != used_key), None
                )
                if fallback is None:
                    raise
                used_key = fallback
                answer, usage, latency_ms = await self._call(
                    used_key,
                    messages,
                    temperature=self.temperature,
                    max_tokens=request.max_tokens or 512,
                )

            score, reason, judge_raw = await self._judge(question, answer)
            total_latency += latency_ms
            if usage:
                total_prompt += usage.prompt_tokens
                total_completion += usage.completion_tokens

            rounds.append(
                {
                    "round": round_index,
                    "model": used_key,
                    "score": round(score, 2),
                    "latency_ms": round(latency_ms, 1),
                    "reason": reason,
                    "judge_raw": judge_raw[:300],
                    "answer": answer,
                }
            )

            if score > best_score:
                best_score = score
                best_answer = answer
                best_reason = reason

            previous_answer = answer
            previous_reason = reason

            if score / 10.0 >= self.threshold:
                break

        if learn:
            q_before, q_after = await self._update_policy(
                task_type,
                used_key,
                quality=max(best_score, 0.0) / 10.0,
                latency_ms=total_latency,
                round_index=len(rounds),
                reason=best_reason,
            )
        else:
            row = await self._policy_row(task_type, used_key)
            q_before = q_after = float(row["q_value"]) if row else 0.5

        metadata = {
            "enabled": True,
            "task_type": task_type,
            "selection_mode": mode,
            "selected_model": used_key,
            "q_before": round(q_before, 4),
            "q_after": round(q_after, 4),
            "policy_updated": learn,
            "threshold": self.threshold,
            "max_rounds": self.max_rounds,
            "judge_model": self.judge_model,
            "rounds": [
                {k: v for k, v in item.items() if k != "answer"} for item in rounds
            ],
            "final_score": round(max(best_score, 0.0), 2),
        }

        return ChatResponse(
            id=f"chatcmpl-rsl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=used_key,
            actual_model=used_key,
            intent_category=task_type,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=best_answer),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
                total_tokens=total_prompt + total_completion,
            ),
            rsl=metadata,
        )

    async def stats(self, limit: int = 50) -> dict:
        rows = await self.db._fetchall(
            """
            SELECT task_type, routing_key, q_value, sample_count,
                   CASE WHEN sample_count > 0 THEN total_quality / sample_count ELSE 0 END AS avg_quality,
                   CASE WHEN sample_count > 0 THEN total_latency_ms / sample_count ELSE 0 END AS avg_latency_ms,
                   updated_at
              FROM rsl_policy
             ORDER BY task_type, q_value DESC
            """
        )
        events = await self.db._fetchall(
            """
            SELECT timestamp, task_type, routing_key, round_index, quality,
                   reward, latency_ms, q_before, q_after, judge_reason
              FROM rsl_events
             ORDER BY id DESC
             LIMIT ?
            """,
            (limit,),
        )
        return {
            "config": {
                "candidates": self.candidates,
                "judge_model": self.judge_model,
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "threshold": self.threshold,
                "max_rounds": self.max_rounds,
            },
            "policy": [dict(row) for row in rows],
            "recent_events": [dict(row) for row in events],
        }

    async def reset(self) -> None:
        await self.db._execute("DELETE FROM rsl_policy")
        await self.db._execute("DELETE FROM rsl_events")
        await self._commit()
        await self.initialize()
