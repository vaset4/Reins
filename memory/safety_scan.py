"""记忆与技能正文落盘前的内容闸门。

判不安全即**整条拒绝**——`memory/writer.py` 拒写、`memory/sediment_sinks.py`
拒收技能正文、`context/memory_recall.py` 把已存的记忆排除出注入。

**授权范围恰好四类**（`SafetyRiskCategory`，spec `backend/error-handling.md`
的 Memory Safe Injection 场景钉死）：提示注入、密钥泄露、命令外泄、隐形字符。
**「绝对路径」刻意不在其中**：路径不是危险内容，而记忆与技能正文天然要写路径
（"配置在 D:\\proj\\config.yml"、"改 memory/writer.py:56"）。把它当 unsafe 会让
四个消费方一起坏——正常记忆拒写、存量记忆隐身、技能正文拒写、每轮沉淀反思中断
（反思 prompt 带 trajectory）。若日后真要治路径，形态是脱敏改写成占位符让内容
仍能落库，不是阻断。

**误报成本比漏报更隐蔽**：召回侧每次重扫每条候选记忆，一条被误判的记忆会从此
对模型隐身，且当前没有任何入口告诉人是哪条、为什么。故密钥值规则一律走
「厂商固定前缀 + 左边界 + 长度下限」，**不做通用高熵检测**——那会打到 commit
sha、哈希名、base64 片段。

与 `runtime.secret_redaction` 是**两个独立所有者**，按该模块 docstring 立的规矩
靠交叉引用与复核对齐、任一侧不得低于另一侧（`tests/test_secret_redaction.py`
的对齐测试守这条）。不合并成一个常量源是因为语义不同：那边是**打码后继续**
（可观测性记录，值被替换但记录仍在），这边是**判不安全后整条拒绝**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_MAX_SCAN_BYTES = 65536


class SafetyRiskCategory(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    CREDENTIAL_LEAK = "credential_leak"
    COMMAND_EXFIL = "command_exfil"
    INVISIBLE_CHAR = "invisible_char"


@dataclass(frozen=True, slots=True)
class SafetyMatch:
    category: SafetyRiskCategory
    pattern_id: str
    snippet: str


@dataclass(frozen=True, slots=True)
class SafetyScanResult:
    is_safe: bool
    matches: tuple[SafetyMatch, ...]


_PATTERNS: dict[SafetyRiskCategory, tuple[tuple[str, re.Pattern[str]], ...]] = {
    SafetyRiskCategory.PROMPT_INJECTION: (
        (
            "ignore_previous_en",
            re.compile(
                r"\bignore\s+(all\s+)?(previous|prior)\s+(rules|instructions)",
                re.IGNORECASE,
            ),
        ),
        (
            "ignore_previous_zh",
            re.compile(r"忽略(之前|以上|所有)(的)?(规则|指令|要求)", re.IGNORECASE),
        ),
        (
            "forget_rules_en",
            re.compile(
                r"\bforget\s+(all\s+)?(rules|instructions|constraints)", re.IGNORECASE
            ),
        ),
        ("you_are_not_zh", re.compile(r"你不是\s*\S+", re.IGNORECASE)),
        ("new_role_en", re.compile(r"\byou\s+are\s+now\s+a\b", re.IGNORECASE)),
        (
            "disregard_en",
            re.compile(
                r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+(rules|instructions)",
                re.IGNORECASE,
            ),
        ),
    ),
    SafetyRiskCategory.CREDENTIAL_LEAK: (
        (
            "env_constant",
            re.compile(
                r"\b(API_KEY|AWS_SECRET|SSH_PRIVATE_KEY|DATABASE_PASSWORD|DB_PASSWORD)\b"
            ),
        ),
        ("dotenv_ref", re.compile(r"\bcat\s+[^\n]*\.env\b", re.IGNORECASE)),
        ("token_assignment", re.compile(r"\btoken\s*=\s*['\"]?\w{20,}", re.IGNORECASE)),
        # 上面三条只认密钥的「名字」（API_KEY= / cat .env / token=），模型在正文里
        # 直接打出一个真密钥值时全部放行。下面六条认「值」，按厂商固定前缀识别。
        # 三条共同约束，缺一条就会误报（代价见模块 docstring 的「误报成本」段）：
        #   1. 左边界 (?<![A-Za-z0-9])——否则 risk-assessment-review 里的 sk- 会命中
        #   2. 长度下限——否则裸前缀 sk- / AKIA 就算密钥
        #   3. AWS/Google 前缀不加 IGNORECASE——否则英文词 akia / aizawa 会命中
        ("openai_key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")),
        ("github_token", re.compile(r"(?<![A-Za-z0-9])gh[porsu]_[A-Za-z0-9]{36,}")),
        ("aws_access_key_id", re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}")),
        (
            "private_key_block",
            re.compile(r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----"),
        ),
        ("slack_token", re.compile(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}")),
        ("google_api_key", re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{35}")),
    ),
    SafetyRiskCategory.COMMAND_EXFIL: (
        ("curl_pipe_sh", re.compile(r"curl[^\n]*\|\s*(sh|bash)", re.IGNORECASE)),
        ("wget_pipe_sh", re.compile(r"wget[^\n]*\|\s*(sh|bash)", re.IGNORECASE)),
        ("dev_tcp", re.compile(r">\s*/dev/tcp/", re.IGNORECASE)),
    ),
    SafetyRiskCategory.INVISIBLE_CHAR: (("zero_width", re.compile(r"[​‌‍﻿‭‮]")),),
}


def scan(text: str) -> SafetyScanResult:
    if not text:
        return SafetyScanResult(is_safe=True, matches=())
    target = text[:_MAX_SCAN_BYTES]
    matches: list[SafetyMatch] = []
    for category, patterns in _PATTERNS.items():
        for pattern_id, regex in patterns:
            match = regex.search(target)
            if match:
                snippet = match.group()[:80]
                matches.append(SafetyMatch(category, pattern_id, snippet))
    return SafetyScanResult(is_safe=not matches, matches=tuple(matches))


__all__ = [
    "SafetyMatch",
    "SafetyRiskCategory",
    "SafetyScanResult",
    "scan",
]
