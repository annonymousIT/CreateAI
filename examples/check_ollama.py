"""Ollama 疎通確認スクリプト。

Week0 ゲートの確証として、OllamaClient で実機に問い合わせて返答を表示する。

使い方:
    1. ``ollama serve`` が起動していること（Windows では Ollama アプリ起動でOK）
    2. ``ollama pull qwen2.5:7b`` 等でモデルを事前に取得しておくこと
    3. ``python examples/check_ollama.py``

モデル名を変えるには:
    ``python examples/check_ollama.py qwen2.5:3b``
"""

from __future__ import annotations

import pathlib
import sys

# 例スクリプトを ``python examples/check_ollama.py`` で直接起動できるよう
# project root を sys.path に通す。本体パッケージ化（pyproject.toml）後は不要。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from createai.llm import OllamaClient  # noqa: E402


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    prompt = "あなたは何ができますか？ 1文で答えてください。"

    print(f"[check_ollama] model = {model}")
    print(f"[check_ollama] prompt = {prompt}")

    with OllamaClient(model=model) as client:
        # 短い応答を期待するシンプルな疎通確認。
        # stop は Week2 の ReAct ループまで使わないが、Week1 で署名を確定しておくため
        # ここでも None を渡しておく（暗黙の動作を契約面で固定する意図）。
        text = client.generate(prompt, stop=None)

    print("[check_ollama] response:")
    print(text.strip() or "(empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
