import fnmatch
import os
import re
from dataclasses import dataclass
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

from runtime.lease import Lease


class Decision(str, Enum):
    ALLOWED = "allowed"
    CONFIRM = "confirm"
    DENY = "deny"


_DEFAULT_DENY_READ = [
    "%USERPROFILE%\\.ssh\\",
    "%USERPROFILE%\\.aws\\",
    "%APPDATA%\\Anthropic",
    "*.pem",
    "*.key",
    ".env",
]


def check_read(path: Path, lease: Lease, *, filtered: bool = False) -> Decision:
    target = resolve_target(path, lease)
    if _matches_deny_read(path, lease, filtered=filtered):
        return Decision.DENY
    if any(_is_under(target, root) for root in _read_roots(lease)):
        return Decision.ALLOWED
    return Decision.CONFIRM


@dataclass(frozen=True)
class ReadBoundary:
    """一次目录捕获的权限快照，复用配置解析但每个目标仍解析实际物理路径。"""

    project_root: Path
    roots: tuple[Path, ...]
    denied: tuple[str, ...]
    sensitive: tuple[str, ...]
    redacted: bool

    @classmethod
    def from_lease(cls, lease: Lease) -> "ReadBoundary":
        """编译当前租约的读取规则；参数：租约；返回：本次遍历可复用的只读规则。"""
        fs = _fs(lease)
        redacted = fs.get("default_sensitive_policy") == "redacted"
        patterns = _strings(fs.get("deny_read"))
        denied = patterns if redacted else patterns or _DEFAULT_DENY_READ
        return cls(
            _project_root(lease) or Path.cwd(),
            tuple(_read_roots(lease)),
            tuple(_prepare_pattern(pattern, lease) for pattern in denied),
            tuple(_prepare_pattern(pattern, lease) for pattern in _DEFAULT_DENY_READ),
            redacted,
        )

    def inspect(self, path: Path) -> tuple[Path, bool, Decision]:
        """同时判断目标、脱敏和读取权限；参数：当前路径；返回：物理路径/敏感性/按脱敏读取的决定。"""
        lexical = path if path.is_absolute() else self.project_root / path
        resolved = lexical.resolve()
        candidates = (lexical, resolved)
        sensitive = self.redacted and any(
            _prepared_pattern_matches(item, rule)
            for rule in self.sensitive
            for item in candidates
        )
        denied = any(
            _prepared_pattern_matches(item, rule)
            for rule in self.denied
            for item in candidates
        )
        decision = (
            Decision.DENY
            if denied
            else (
                Decision.ALLOWED
                if any(_is_under(resolved, root) for root in self.roots)
                else Decision.CONFIRM
            )
        )
        return resolved, sensitive, decision


def _prepare_pattern(pattern: str, lease: Lease) -> str:
    """冻结配置规则的展开结果；参数：原规则/租约；返回：同一匹配语义的规范规则。"""
    expanded = _expand(pattern, lease)
    return (
        str(Path(expanded).resolve())
        if _has_path_part(expanded) and not any(char in expanded for char in "*?")
        else expanded
    )


def _prepared_pattern_matches(path: Path, pattern: str) -> bool:
    """匹配已展开的规则，不重复查询相同配置路径；参数：目标/规则；返回：是否命中。"""
    if not pattern:
        return False
    if not _has_path_part(pattern):
        return fnmatch.fnmatch(_norm(path.name), _norm(pattern))
    if "*" in pattern or "?" in pattern:
        return fnmatch.fnmatch(_norm(path), _norm(pattern))
    return _is_under(path, Path(pattern))


def check_write(path: Path, lease: Lease, *, filtered: bool = False) -> Decision:
    target = resolve_target(path, lease)
    if _matches_deny_write(path, lease, filtered=filtered):
        return Decision.DENY

    workspace_base = _workspace_base(lease)
    if workspace_base is not None and _is_under(target, workspace_base):
        task_dir = workspace_base / lease.task_id
        return Decision.ALLOWED if _is_under(target, task_dir) else Decision.DENY

    if any(_is_under(target, root) for root in _write_roots(lease)):
        return Decision.ALLOWED

    if _is_under(target, _data_root()):
        return Decision.ALLOWED

    project_root = _project_root(lease)
    if project_root is not None and _is_under(target, project_root):
        return Decision.CONFIRM

    return Decision.DENY


def task_workspace(lease: Lease) -> Path | None:
    # exec 工具的受控工作目录：lease workspace 下以 task_id 命名的子目录。
    # 与 check_write 的 workspace 限定同源；exec 入口默认 cwd 落于此。
    workspace_base = _workspace_base(lease)
    if workspace_base is None:
        return None
    return workspace_base / lease.task_id


def classify_exec_targets(text: str, lease: Lease) -> Decision:
    # 尽力而为的字符串扫描，非 OS 沙箱：从命令/代码文本里挑出"像路径"的 token，套用与结构化
    # fs 工具同一套 deny 名单 + roots 判定（不另起一套）。挡 `cat .env` / `open(".env")` /
    # 越界绝对路径这类手滑与朴素注入；挡不住刻意混淆（base64 路径、env 间接拼接、下载再执行）。
    decision = Decision.ALLOWED
    for token in _exec_path_tokens(text):
        verdict = _classify_exec_token(token, lease)
        if verdict is Decision.DENY:
            return Decision.DENY
        if verdict is Decision.CONFIRM:
            decision = Decision.CONFIRM
    return decision


def _classify_exec_token(token: str, lease: Lease) -> Decision:
    target = resolve_target(Path(token), lease)
    if _matches_deny_read(Path(token), lease):
        if (
            lease.trigger != "cron"
            and uses_redacted_files(Path(token), lease)
            and not _matches_deny_read(Path(token), lease, filtered=True)
        ):
            return Decision.CONFIRM
        return Decision.DENY
    if _exec_token_out_of_bounds(token, target, lease):
        return Decision.CONFIRM
    return Decision.ALLOWED


def _exec_token_out_of_bounds(token: str, target: Path, lease: Lease) -> bool:
    parsed = Path(token)
    if not (parsed.is_absolute() or ".." in parsed.parts):
        return False
    roots = _read_roots(lease) + _write_roots(lease)
    return not any(_is_under(target, root) for root in roots)


_EXEC_QUOTED = re.compile(r"""['"]([^'"]+)['"]""")
_EXEC_SPLIT = re.compile(r"[\s,;|&()<>]+")


def _exec_path_tokens(text: str) -> list[str]:
    tokens = list(_EXEC_QUOTED.findall(text))
    for raw in _EXEC_SPLIT.split(text):
        candidate = raw.strip().strip("'\"")
        if candidate and _looks_like_path(candidate):
            tokens.append(candidate)
    return tokens


def execution_target_paths(text: str, cwd: Path) -> tuple[Path, ...]:
    """提取命令明确点名的跨目录目标；参数：命令正文/实际工作目录；返回：物理路径，不授予权限。"""
    targets = set()
    for token in _exec_path_tokens(text):
        path = Path(token)
        if path.is_absolute() or ".." in path.parts:
            targets.add((path if path.is_absolute() else cwd / path).resolve())
    return tuple(sorted(targets))


def _looks_like_path(token: str) -> bool:
    if token in {".", ".."} or token.startswith("-"):
        return False
    if any(sep in token for sep in ("/", "\\", ":")):
        return True
    if token.startswith(("~", ".")) and len(token) > 1:
        return True
    return "." in token


def uses_redacted_files(path: Path, lease: Lease) -> bool:
    """识别可走脱敏通道的默认敏感路径；传参：路径、权限；返回：是否具有明确策略来源。"""
    return _fs(lease).get("default_sensitive_policy") == "redacted" and any(
        _matches_pattern(path, pattern, lease) for pattern in _DEFAULT_DENY_READ
    )


def _matches_deny_read(path: Path, lease: Lease, *, filtered: bool = False) -> bool:
    fs = _fs(lease)
    if fs.get("default_sensitive_policy") == "redacted":
        explicit = any(
            _matches_pattern(path, pattern, lease)
            for pattern in _strings(fs.get("deny_read"))
        )
        return explicit or (not filtered and uses_redacted_files(path, lease))
    patterns = _strings(fs.get("deny_read")) or _DEFAULT_DENY_READ
    return any(_matches_pattern(path, pattern, lease) for pattern in patterns)


def _matches_deny_write(path: Path, lease: Lease, *, filtered: bool = False) -> bool:
    fs = _fs(lease)
    if fs.get("default_sensitive_policy") == "redacted":
        patterns = _strings(fs.get("deny_read")) + _strings(fs.get("deny_write"))
        explicit = any(_matches_pattern(path, pattern, lease) for pattern in patterns)
        return explicit or (not filtered and uses_redacted_files(path, lease))
    if "deny_write" in fs:
        patterns = _strings(fs.get("deny_write"))
    else:
        patterns = _strings(fs.get("deny_read")) or _DEFAULT_DENY_READ
    return any(_matches_pattern(path, pattern, lease) for pattern in patterns)


def _matches_pattern(path: Path, pattern: str, lease: Lease) -> bool:
    """同时核对调用路径和链接目标，保留两侧禁令；传参：路径、模式、权限；返回：任一匹配。"""
    lexical = (
        path if path.is_absolute() else (_project_root(lease) or Path.cwd()) / path
    )
    return any(
        _matches_absolute_pattern(candidate, pattern, lease)
        for candidate in (lexical, lexical.resolve())
    )


def _matches_absolute_pattern(path: Path, pattern: str, lease: Lease) -> bool:
    """核对一份绝对路径与原有名单模式；传参：绝对路径、模式、权限；返回：匹配结果。"""
    return _prepared_pattern_matches(path, _prepare_pattern(pattern, lease))


def _read_roots(lease: Lease) -> list[Path]:
    fs = _fs(lease)
    configured_roots = _strings(fs.get("read"))
    if configured_roots:
        return [
            Path(_expand(item, lease)).resolve()
            for item in configured_roots
            if "<project>" not in item or _project_root(lease) is not None
        ]

    roots = [_data_root()]
    project_root = _project_root(lease)
    workspace = _workspace_base(lease)
    if workspace is not None:
        roots.append(workspace)
    if project_root is not None:
        roots.append(project_root)
    return roots


def _write_roots(lease: Lease) -> list[Path]:
    fs = _fs(lease)
    return [
        Path(_expand(item, lease)).resolve()
        for item in _strings(fs.get("write"))
        if "<project>" not in item or _project_root(lease) is not None
    ]


def resolve_target(path: Path, lease: Lease) -> Path:
    """按同一项目根解析权限校验与审批资源；传参：路径/租约；返回：实际规范化路径。"""
    if path.is_absolute():
        return path.resolve()
    project_root = _project_root(lease)
    if project_root is not None:
        return (project_root / path).resolve()
    return path.resolve()


def _workspace_base(lease: Lease) -> Path | None:
    fs = _fs(lease)
    for key in ("workspace_root", "workspace"):
        value = fs.get(key)
        if isinstance(value, str):
            return Path(_expand(value, lease)).resolve()
    project_root = _project_root(lease)
    if project_root is not None:
        return (project_root / ".reins" / "workspace").resolve()
    return None


def _project_root(lease: Lease) -> Path | None:
    fs = _fs(lease)
    value = fs.get("project_root")
    if isinstance(value, str):
        return Path(_expand(value, lease)).resolve()
    data_root = _data_root()
    for item in _strings(fs.get("read")):
        if "<project>" in item:
            continue
        root = Path(_expand(item, lease)).resolve()
        if root == data_root or root.name == "workspace":
            continue
        return root
    return None


def _fs(lease: Lease) -> Mapping[str, object]:
    value = lease.capabilities.get("fs")
    return value if isinstance(value, Mapping) else {}


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _expand(value: str, lease: Lease) -> str:
    project_root = _project_root_raw(lease)
    if project_root is not None:
        value = value.replace("<project>", str(project_root))
    return os.path.expandvars(os.path.expanduser(value))


def _project_root_raw(lease: Lease) -> Path | None:
    value = _fs(lease).get("project_root")
    return Path(value) if isinstance(value, str) else None


def _data_root() -> Path:
    return (Path.home() / ".reins" / "data").resolve()


def _is_under(path: Path, root: Path) -> bool:
    path_text = _norm(path)
    root_text = _norm(root).rstrip("/")
    return path_text == root_text or path_text.startswith(root_text + "/")


def _has_path_part(value: str) -> bool:
    return "/" in value or "\\" in value or ":" in value


def _norm(path: Path | str) -> str:
    return str(path).replace("\\", "/").casefold().rstrip("/")
