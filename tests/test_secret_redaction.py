"""Cross-store secret redaction consistency (PRD R1/R3).

Both ``run_evidence`` and ``run_facts`` must route secret protection through the
single ``runtime.secret_redaction`` owner so that neither store leaks a secret
the other store would have caught. The owner merges two layers:

- key-based: a secret-looking key (``api_key``, ``authorization``, ``cookie`` ...)
  has its whole value replaced with ``<redacted>``.
- value-pattern: a secret-looking value (``sk-...``, ``Bearer ...``) embedded in
  an otherwise innocuous key/string is masked in place.

R3 (only stricter, never looser): a payload that mixes both must come out with
both layers applied in either store, with no plaintext secret surviving.
"""

from __future__ import annotations

import json
from pathlib import Path

from memory.safety_scan import scan as safety_scan
from runtime.run_evidence import RunEvidenceStore
from runtime.run_facts import RunFactStore
from runtime.secret_redaction import REDACTED, redact_secret_text, redact_value

_SECRET_KEY_VALUE = "supersecretkeyvalue123456"
_SECRET_TOKEN_IN_TEXT = "sk-abc12345678901234567890"
_BEARER_IN_TEXT = "Authorization: Bearer abcdefghijklmnopqrstuv"


def _mixed_payload() -> dict[str, object]:
    return {
        "api_key": _SECRET_KEY_VALUE,
        "cookie": "session=abc123",
        "message": f"call failed using {_SECRET_TOKEN_IN_TEXT}",
        "header_line": _BEARER_IN_TEXT,
        "total_tokens": 142,
    }


def _assert_no_plaintext_secret(blob: str) -> None:
    assert _SECRET_KEY_VALUE not in blob
    assert _SECRET_TOKEN_IN_TEXT not in blob
    assert "abcdefghijklmnopqrstuv" not in blob


def test_redact_value_applies_both_layers() -> None:
    redacted = redact_value(_mixed_payload())
    assert isinstance(redacted, dict)
    # key-based: whole value replaced
    assert redacted["api_key"] == "<redacted>"
    assert redacted["cookie"] == "<redacted>"
    # value-pattern: secret token masked in place inside a non-secret key
    assert _SECRET_TOKEN_IN_TEXT not in str(redacted["message"])
    assert "Bearer <redacted>" in str(redacted["header_line"])
    # innocuous values preserved
    assert redacted["total_tokens"] == 142
    _assert_no_plaintext_secret(json.dumps(redacted))


def test_redact_secret_text_masks_value_patterns() -> None:
    text = f"used {_SECRET_TOKEN_IN_TEXT} and {_BEARER_IN_TEXT}"
    redacted = redact_secret_text(text)
    _assert_no_plaintext_secret(redacted)
    assert "Bearer <redacted>" in redacted


# 真实密钥样本：两个所有者都必须认得。只放确凿的厂商形状，不放任意字符串——
# 既有 sk- 规则缺左边界会在普通散文上假命中（见 test_existing_sk_rule_lacks_left_boundary），
# 拿垃圾输入跑对齐会让断言失败在一个假命中上。
_REAL_CREDENTIAL_SAMPLES: tuple[tuple[str, str], ...] = (
    ("openai", "sk-proj-" + "a" * 24),
    ("github", "ghp_" + "A" * 36),
    ("aws", "AKIAIOSFODNN7EXAMPLE"),
    ("private_key", "-----BEGIN RSA PRIVATE KEY-----"),
    ("slack", "xoxb-123456789012-" + "a" * 24),
    ("google", "AIza" + "b" * 35),
)


def test_redact_masks_all_vendor_credential_shapes() -> None:
    """run_facts / run_evidence 落盘前要认全同一批厂商前缀。

    补齐前只有 sk- 与 Bearer 两条，故模型读到一个含 GitHub token 的文件后，
    该 token 会明文进 run facts 落盘。
    """
    for label, sample in _REAL_CREDENTIAL_SAMPLES:
        redacted = redact_secret_text(f"value is {sample}")
        assert sample not in redacted, f"{label} 未被打码: {redacted}"
        assert REDACTED in redacted, label


def test_safety_scan_never_weaker_than_redaction() -> None:
    """两个所有者的对齐纪律第一次被机器守住。

    `runtime/secret_redaction.py` docstring 立的规矩是「任一侧不得低于另一侧」，
    但此前只靠人工复核。这条断言 redaction 认得的真密钥形状，safety_scan 一律
    判不安全（即阻断闸门不低于打码器）。反向不作要求——redaction 是打码器，
    不承担 GitHub token 的阻断语义。
    """
    for label, sample in _REAL_CREDENTIAL_SAMPLES:
        masked = REDACTED in redact_secret_text(sample)
        blocked = not safety_scan(sample).is_safe
        assert masked and blocked, (
            f"{label} 在两侧不一致：redaction 打码={masked}、safety_scan 阻断={blocked}"
        )


def test_existing_sk_rule_lacks_left_boundary() -> None:
    """留档一个本卡刻意不修的既有误报（`secret_redaction.py` 的 sk- 规则）。

    该规则没有左边界，故 `risk-assessment-review` 被打成 `ri<redacted>`。
    不是泄密（方向是过度打码），但在破坏可观测性记录的可读性。
    本卡不修的理由：加左边界属于**放宽**一条现役脱敏规则，而该模块自立契约是
    "only stricter, never looser"，放宽需显式授权，不能在补规则的卡里顺手做。
    本卡新增六条规则均自带左边界，未扩大影响面（见
    `tests/test_memory_safety_scan.py::test_left_boundary_protects_ordinary_prose`）。
    """
    assert REDACTED in redact_secret_text("do a risk-assessment-review first")
    # 同一输入在阻断侧不误判——两侧代价不同，故边界写法刻意不一致
    assert safety_scan("do a risk-assessment-review first").is_safe


def test_run_evidence_and_run_facts_redact_consistently(tmp_path: Path) -> None:
    """当前证据与事实原件都保存脱敏内容，读取结果保持一致；参数：隔离目录；返回：无。"""
    evidence = RunEvidenceStore(tmp_path)
    facts = RunFactStore(tmp_path)
    payload = _mixed_payload()

    evidence_ref = evidence.write_record(
        session_id="session-x",
        run_id="run-x",
        kind="model_request",
        source_id="request-1",
        payload=payload,
    )
    facts.append(
        {
            "event": "llm:response",
            "ts": "2026-06-02T00:00:00Z",
            "session_id": "session-x",
            "run_id": "run-x",
            "task_id": "task-x",
            **payload,
        }
    )

    evidence_payload = evidence.read_reference(evidence_ref)
    assert evidence_payload is not None
    evidence_blob = json.dumps(evidence_payload)
    fact_rows = facts.read_run("run-x")
    fact_blob = json.dumps(fact_rows)

    # Neither store leaks a secret the other store would have caught (R1/R3).
    _assert_no_plaintext_secret(evidence_blob)
    _assert_no_plaintext_secret(fact_blob)
    # 1. 【秘密保护】【持久原件】读取时脱敏不足以证明安全，实际保存的两类事件也不能保留明文
    for original in tmp_path.rglob("events.jsonl"):
        _assert_no_plaintext_secret(original.read_text(encoding="utf-8"))

    # Both stores apply key-based redaction to secret keys.
    assert evidence_payload["api_key"] == "<redacted>"
    assert evidence_payload["cookie"] == "<redacted>"
    fact = next(row for row in fact_rows if row.get("event") == "llm:response")
    assert fact["api_key"] == "<redacted>"
    assert fact["cookie"] == "<redacted>"

    # Both stores apply value-pattern redaction to secrets under innocuous keys.
    assert _SECRET_TOKEN_IN_TEXT not in str(evidence_payload["message"])
    assert _SECRET_TOKEN_IN_TEXT not in str(fact["message"])
