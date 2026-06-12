"""Tool / ToolRegistry — エージェントが使う能力の登録・検索・実行。

契約は ``docs/interface.md`` §2-§3 を参照。設計の背景は ``docs/design.md`` §4.3。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def render_for_prompt(self) -> str:
        lines: list[str] = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            lines.append(f"  parameters: {json.dumps(tool.parameters, ensure_ascii=False)}")
        return "\n".join(lines)

    def invoke(self, name: str, arguments: dict) -> str:
        # 契約: invoke は決して例外を投げない。失敗は Observation 文字列に変換する。
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self._tools.keys()) or "(none)"
            return f"ERROR: unknown tool '{name}'. available: {available}"
        try:
            result = tool.func(**arguments)
        except TypeError as exc:
            return f"ERROR: invalid arguments for '{name}': {exc}"
        except Exception as exc:
            return f"ERROR: tool '{name}' failed: {type(exc).__name__}: {exc}"
        if not isinstance(result, str):
            return f"ERROR: tool '{name}' returned non-str ({type(result).__name__})"
        return result
