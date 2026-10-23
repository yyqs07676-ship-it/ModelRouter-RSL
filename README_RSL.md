# ModelRouter + RSL（Recursive Self-Learning）

在课程提供的 [ModelRouterCourse](https://github.com/NewYearEve-Godsheep/ModelRouterCourse) 基础上加入 RSL 闭环：
**任务分类 → 模型选择 → 生成 → 质量评估 → 递归修订 → Q 值更新**。

系统学习的是「哪类任务该交给哪个模型」，**不修改任何大模型的权重**。

## 结果速览

| 指标 | 冷启动（未学习） | 学习后（冻结策略评估） |
| --- | --- | --- |
| 平均得分（6 道留出题） | **8.67** | **10.00** |
| 路由选择 | 全部 `qwen-small`（6/6） | 全部 `qwen-strong`（6/6） |

最终 Q 表（`reasoning` 任务）：

| 模型 | Q 值 | 采样次数 | 平均质量 |
| --- | --- | --- | --- |
| `qwen-strong`（qwen2.5:3b） | **0.7509** | 30 | 0.920 |
| `qwen-small`（qwen2.5:0.5b） | 0.6191 | 18 | 0.822 |

> 完整实验数据见 [`results/rsl_experiment.json`](results/rsl_experiment.json) 与
> [`results/rsl_experiment.md`](results/rsl_experiment.md)。

## 架构

```
POST /v1/chat/completions  (model = "rsl-auto")
        │
        ▼
┌───────────────────┐
│ 任务分类           │  关键词分类：coding / translation / reasoning / writing / general
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ ε-greedy 选模型    │  Q[task_type][routing_key]，未探索时乐观初始化
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 生成答案           │  复用 ModelRouter 的 provider 适配器
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 质量评估           │  可机判题 → 确定性校验器；开放题 → LLM 裁判
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 递归修订（可选）    │  分数低于阈值时，把评审意见反馈给同一模型重写
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ 策略更新           │  Q ← Q + α(reward − Q)，写入 SQLite
└───────────────────┘
```

## 快速开始

### 前置条件

- Python 3.10+（本项目在 Python 3.13.15 上验证）
- [Ollama](https://ollama.com/) 已安装并运行在 `http://127.0.0.1:11434`

### 拉取模型

```bash
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:3b
```

### 安装与启动

```bash
python -m pip install -r requirements.txt
python -m app.main
```

默认监听 `http://127.0.0.1:8000`。健康检查：

```bash
curl http://127.0.0.1:8000/health
```

应看到 `"status": "healthy"`、`"providers": 1`、`"models": 2`、`"rsl": true`。

> 仓库中的 `.env` 只包含本地开发用的占位密钥（`dev-router-key` / `dev-admin-key`），
> 不包含任何真实凭证。生产环境请替换。

## API

所有接口使用 `Authorization: Bearer dev-router-key`。

### 1. RSL 聊天（学习开关可控）

```bash
curl -X POST http://127.0.0.1:8000/v1/rsl/chat \
  -H "Authorization: Bearer dev-router-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"rsl-auto","messages":[{"role":"user","content":"计算 48 和 18 的最大公约数，并说明步骤。"}]}'
```

查询参数：

- `explore=true|false`：是否允许 ε-greedy 探索
- `learn=true|false`：是否把本次结果写回 Q 表（评估时应为 `false`）

响应中包含 `rsl` 元数据：

```json
{
  "rsl": {
    "task_type": "reasoning",
    "selection_mode": "exploit",
    "selected_model": "qwen-strong",
    "q_before": 0.6191,
    "q_after": 0.7509,
    "policy_updated": true,
    "final_score": 10.0,
    "rounds": [{"round": 1, "score": 10.0, "reason": "规则校验通过：最大公约数 = 6"}]
  }
}
```

### 2. 查看学到的策略

```bash
curl http://127.0.0.1:8000/v1/rsl/stats -H "Authorization: Bearer dev-router-key"
```

### 3. 重置策略

```bash
curl -X POST http://127.0.0.1:8000/v1/rsl/reset -H "Authorization: Bearer dev-router-key"
```

### 4. 主路由入口

`POST /v1/chat/completions` 在 `model="rsl-auto"` 时走 RSL 策略；其他模型名保持原有行为不变。

## RSL 算法

### 任务分类

关键词分类器（`app/core/rsl.py:classify_task`），可解释、无需额外模型：

| 类别 | 触发词示例 |
| --- | --- |
| coding | 代码、函数、算法、报错、python |
| translation | 翻译、中译英、translate |
| reasoning | 计算、证明、推理、最大公约数 |
| writing | 写一篇、作文、润色、摘要 |
| general | 兜底 |

### 选择策略

- 以概率 ε 随机探索，以 1−ε 选择当前 Q 值最高的模型
- Q 值相同时优先选择采样次数更少的模型（乐观初始化）
- 冷启动策略：所有 Q=0.5，选择顺序中的第一个候选（成本更低的 `qwen-small`）

### 奖励

```
quality  = score / 10
latency  = min(total_latency_ms / 30000, 1)
reward   = max(0, min(1, quality − 0.1 × latency))
Q_new    = Q_old + α × (reward − Q_old)
```

默认 `α = 0.25`、`ε = 0.25`、阈值 `0.75`、最多 2 轮修订。

### 质量信号：为什么需要确定性校验

小参数模型自评不可靠。实验中 1.5B 裁判曾把「48 和 18 的最大公约数 = 3」判为 7 分（正确答案是 6）。
因此实现了 `app/core/rsl_verifier.py`：

- 最大公约数 / 加法 / 乘法 → 直接用 Python 计算标准答案并核对结论
- 其他任务 → 回退到 LLM 裁判，并在无法解析时重试一次「只输出数字」的提示

## 实验

复现命令：

```bash
python scripts/run_rsl_experiment.py
```

协议：

1. 重置策略（所有 Q = 0.5）
2. **冷启动评估**：6 道留出题，`explore=false, learn=false`
3. **学习阶段**：24 道训练题，固定种子打乱，跑 2 轮共 48 次请求（`ε=0.4, α=0.3`）
4. **学习后评估**：同一组留出题，`explore=false, learn=false`（冻结策略，避免评估过程反过来改策略）

结果：

| 阶段 | 平均得分 | 模型选择 |
| --- | --- | --- |
| 冷启动 | 8.67 | qwen-small ×6 |
| 学习后 | 10.00 | qwen-strong ×6 |

## 测试

```bash
python -m pytest tests/test_rsl.py -q
```

覆盖：任务分类、裁判输出解析、确定性校验器（含「结论数字提取」回归用例）、ε-greedy 选择与平票规则。

## 目录结构

```
app/
├── api/
│   ├── chat.py            # 主入口；model=rsl-auto 时转交 RSL
│   └── rsl.py             # /v1/rsl/chat /v1/rsl/stats /v1/rsl/reset
├── core/
│   ├── rsl.py             # RSL 引擎：分类、选择、递归、Q 更新
│   ├── rsl_verifier.py    # 确定性答案校验
│   └── ...
└── providers/             # 原有供应商适配器（复用）
config/providers.yaml      # 本地 Ollama 注册为 OpenAI 兼容供应商
scripts/
├── run_rsl_experiment.py  # 冷启动 vs 学习后实验
├── probe_models.py        # 模型能力探测
└── probe_objective.py     # 客观任务探测
tests/test_rsl.py
results/                   # 实验产物（JSON + Markdown）
```

## 已知局限

1. **模型能力有限**：0.5B / 3B 在最大公约数等多步推理上都会犯错，单次结果波动明显；实验用 `temperature=0` 固定采样以保证可复现。
2. **同模型自评不可靠**：LLM 裁判可能给错误答案高分，因此可机判题一律走规则校验器。
3. **奖励信号稀疏**：校验器只给 10 / 2 两档，Q 值收敛慢；接入更细粒度评分（0–100）会更稳定。
4. **单一任务类别**：实验集中在 `reasoning`；代码、翻译、写作类任务需要更可靠的自动评测才能得到同样可信的曲线。
5. **不做权重更新**：本项目的 RSL 是策略/经验层面的学习，不是 LoRA 微调。

## 与上游的关系

保留上游的 OpenAI 兼容接口、多供应商适配、routing profile、cascade、ensemble 与管理接口；
RSL 只新增策略层与质量评估层，不改变原有模型转发逻辑。
