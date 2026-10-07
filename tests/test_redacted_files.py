"""验证配置结构可编辑、原秘密逐字节保留及显式替换边界。

作者：xxx
时间：2026-09-24 20:30:00
"""

import re

import pytest

from tools.redacted_files import RedactedFiles

TOKEN = re.compile(r"<redacted:[0-9a-f]{32}>")


@pytest.mark.parametrize("format", ["dotenv", "ini"])
def test_redacted_config_roundtrip_preserves_secret_bytes_and_layout(tmp_path, format):
    """修改普通端口时原秘密、引号、BOM和换行保持原字节；传参：目录与格式；返回：无。"""
    path = tmp_path / (".env" if format == "dotenv" else "credentials")
    prefix = b"\xef\xbb\xbf" + (b"[default]\r\n" if format == "ini" else b"")
    original = (
        prefix
        + b"# credential is SAME_SECRET\r\nPORT=3000\r\nTOKEN='SAME_SECRET'\r\nPASSWORD=SAME_SECRET\r\nDATABASE_URL=postgres://u:p@host/db\r\n"
    )
    path.write_bytes(original)
    files = RedactedFiles()
    view = files.read(path, session_id="session")
    assert "SAME_SECRET" not in str(view) and "postgres://" not in str(view)
    assert "PORT=3000" in view["content"]
    tokens = TOKEN.findall(view["content"])
    assert len(tokens) == len(set(tokens)) == 4
    edit = files.prepare(
        "file_write",
        {
            "view_id": view["meta"]["view_id"],
            "content": view["content"].replace("3000", "4000"),
        },
        session_id="session",
        path=path,
    )
    result = files.publish(edit)
    assert path.read_bytes() == original.replace(b"3000", b"4000")
    assert result["meta"]["previous_sha256"] == view["meta"]["content_sha256"]


def test_multiline_dotenv_and_inline_comments_never_leak(tmp_path):
    """多行值、转义和行尾注释都由同一完整快照保护；传参：目录；返回：无。"""
    path = tmp_path / ".env"
    original = b'export TOKEN="first-line\nsecond\\"line" # first-line\nPORT=3000 # port note\nEMPTY=\n'
    path.write_bytes(original)
    files = RedactedFiles()
    view = files.read(path, session_id="session")
    assert "first-line" not in view["content"] and "port note" not in view["content"]
    edit = files.prepare(
        "file_patch",
        {
            "view_id": view["meta"]["view_id"],
            "old_text": "PORT=3000",
            "new_text": "PORT=5000",
        },
        session_id="session",
        path=path,
    )
    files.publish(edit)
    assert path.read_bytes() == original.replace(b"3000", b"5000")


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "unknown", "other_session", "changed_file"]
)
def test_invalid_or_stale_view_never_overwrites_secret(tmp_path, failure):
    """编号和版本不符时没有写入副作用，错误不包含原秘密；传参：目录与错误类型；返回：无。"""
    path = tmp_path / ".env"
    original = b"TOKEN=PRIVATE_ORIGINAL\nPORT=3000\n"
    path.write_bytes(original)
    files = RedactedFiles()
    view = files.read(path, session_id="session")
    token = TOKEN.search(view["content"]).group()
    text = view["content"]
    if failure == "missing":
        text = text.replace(token, "")
    elif failure == "duplicate":
        text += f"EXTRA={token}\n"
    elif failure == "unknown":
        text = text.replace(token, "<redacted:" + "0" * 32 + ">")
    elif failure == "changed_file":
        path.write_bytes(original.replace(b"3000", b"9000"))
    before = path.read_bytes()
    with pytest.raises(ValueError) as error:
        edit = files.prepare(
            "file_write",
            {"view_id": view["meta"]["view_id"], "content": text},
            session_id="other" if failure == "other_session" else "session",
            path=path,
        )
        files.publish(edit)
    assert "PRIVATE_ORIGINAL" not in str(error.value) and path.read_bytes() == before


def test_explicit_secret_replacement_uses_protected_input_reference(tmp_path):
    """新值先进入宿主暂存，编辑参数只保存引用；传参：目录；返回：无。"""
    path = tmp_path / ".env"
    path.write_bytes(b"TOKEN=OLD_SECRET\nPASSWORD=OLD_SECRET\n")
    files = RedactedFiles()
    view = files.read(path, session_id="session")
    first, second = TOKEN.findall(view["content"])
    args = files.protect_arguments(
        {
            "view_id": view["meta"]["view_id"],
            "content": view["content"],
            "secret_replacements": {first: "NEW_SECRET"},
        },
        session_id="session",
        request_id="request",
        call_id="call",
    )
    assert "NEW_SECRET" not in str(args)
    edit = files.prepare("file_write", args, session_id="session", path=path)
    assert edit.replacement_count == 1 and "NEW_SECRET" not in repr(edit)
    files.publish(edit)
    assert path.read_bytes() == b'TOKEN="NEW_SECRET"\nPASSWORD=OLD_SECRET\n'
    assert first != second


def test_same_value_positions_are_independent_and_unchanged_fields_reuse_tokens(
    tmp_path,
):
    """按字段出现位置复用编号，值相同不共享映射；传参：目录；返回：无。"""
    path = tmp_path / ".env"
    path.write_text("TOKEN=shared\nPASSWORD=shared\nPORT=3000\n", encoding="utf-8")
    files = RedactedFiles()
    first = files.read(path, session_id="session")
    path.write_text("TOKEN=shared\nPASSWORD=changed\nPORT=4000\n", encoding="utf-8")
    second = files.read(path, session_id="session")
    old, new = TOKEN.findall(first["content"]), TOKEN.findall(second["content"])
    assert old[0] == new[0] and old[1] != new[1] and old[0] != old[1]
    assert first["meta"]["view_id"] != second["meta"]["view_id"]


@pytest.mark.parametrize(
    ("name", "content"),
    [
        (".env", b'TOKEN="unfinished secret'),
        (".env", b"TOKEN=SECRET\nunrecognized text\n"),
        ("credentials", b"TOKEN=SECRET\n"),
        ("config.json", b'{"token":"SECRET"}'),
        ("key.pem", b"-----BEGIN PRIVATE KEY-----\nSECRET\n-----END PRIVATE KEY-----"),
    ],
)
def test_unsupported_material_only_returns_metadata(tmp_path, name, content):
    """不能完整解析或私钥材料只返回元信息；传参：目录、文件名和原字节；返回：无。"""
    path = tmp_path / name
    path.write_bytes(content)
    view = RedactedFiles().read(path, session_id="session")
    assert view["meta"]["metadata_only"] is True and "SECRET" not in str(view)


def test_partial_view_cannot_replace_whole_file_but_can_patch(tmp_path):
    """分页不完整时禁止整写，补丁仍在完整基准内修改；传参：目录；返回：无。"""
    path = tmp_path / ".env"
    path.write_text("PORT=3000\nTOKEN=PRIVATE\n", encoding="utf-8")
    files = RedactedFiles()
    page = files.read(path, session_id="session", max_chars=10)
    assert not page["meta"]["output_complete"] and page["meta"]["next_offset"] == 10
    with pytest.raises(ValueError, match="incomplete"):
        files.prepare(
            "file_write",
            {"view_id": page["meta"]["view_id"], "content": page["content"]},
            session_id="session",
            path=path,
        )
    edit = files.prepare(
        "file_patch",
        {
            "view_id": page["meta"]["view_id"],
            "old_text": "PORT=3000",
            "new_text": "PORT=4000",
        },
        session_id="session",
        path=path,
    )
    files.publish(edit)
    assert path.read_text() == "PORT=4000\nTOKEN=PRIVATE\n"


def test_explicit_replacement_accepts_new_value_at_original_position(tmp_path):
    """明确关联的替换允许正文放入新值，暂存后仍能确定原位置；传参：目录；返回：无。"""
    path = tmp_path / ".env"
    path.write_text("TOKEN=PRIVATE\n", encoding="utf-8")
    files = RedactedFiles()
    view = files.read(path, session_id="session")
    token = TOKEN.search(view["content"]).group()
    args = files.protect_arguments(
        {
            "view_id": view["meta"]["view_id"],
            "content": view["content"].replace(token, '"NEW_PRIVATE"'),
            "secret_replacements": {token: "NEW_PRIVATE"},
        },
        session_id="session",
        request_id="request",
        call_id="call",
    )
    assert "NEW_PRIVATE" not in str(args)
    files.publish(files.prepare("file_write", args, session_id="session", path=path))
    assert path.read_text() == 'TOKEN="NEW_PRIVATE"\n'


def test_ini_continuation_after_blank_and_comment_is_not_a_new_field(tmp_path):
    """INI值跨过空行和注释继续缩进时整体保密并原样往返；传参：目录；返回：无。"""
    path = tmp_path / "credentials"
    raw = b"[default]\nTOKEN=PRIVATE\n\n# NOTE_PRIVATE\n    continued PRIVATE\nPORT=3000\n"
    path.write_bytes(raw)
    files = RedactedFiles()
    view = files.read(path, session_id="session")
    assert not view["meta"]["metadata_only"] and "PRIVATE" not in str(view)
    files.publish(
        files.prepare(
            "file_patch",
            {
                "view_id": view["meta"]["view_id"],
                "old_text": "PORT=3000",
                "new_text": "PORT=4000",
            },
            session_id="session",
            path=path,
        )
    )
    assert path.read_bytes() == raw.replace(b"3000", b"4000")


def test_publish_rechecks_authority_inside_file_lock(tmp_path):
    """排队后撤权在真实发布线程内阻止写入；传参：目录；返回：无。"""
    path = tmp_path / ".env"
    path.write_bytes(b"TOKEN=PRIVATE\nPORT=3000\n")
    files = RedactedFiles()
    view = files.read(path, session_id="session")
    edit = files.prepare(
        "file_patch",
        {
            "view_id": view["meta"]["view_id"],
            "old_text": "PORT=3000",
            "new_text": "PORT=4000",
        },
        session_id="session",
        path=path,
    )

    def revoked():
        """模拟锁内最终授权复核发现撤销；传参：无；返回：无，抛明确错误。"""
        raise PermissionError("authorization revoked")

    with pytest.raises(PermissionError, match="revoked"):
        files.publish(edit, validate=revoked)
    assert path.read_bytes() == b"TOKEN=PRIVATE\nPORT=3000\n"
