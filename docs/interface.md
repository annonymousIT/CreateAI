# CreateAI 内部 I/F 契約

> Phase 1 開発において、コンポーネント間で**両者合意済み**として固定するインターフェース契約。
> 本ファイルを変更する場合は、両者の合意を必須とする（Sho / Yuki）。
> 設計の背景は [`design.md`](design.md) を、選択理由は [`tradeoffs.md`](tradeoffs.md) を参照。

---

## 1. `LLMClient` — LLM 抽象

### 署名

```python
from typing import Protocol

class LLMClient(Protocol):
    def generate(self, prompt: str, *, stop: list[str] | None = None) -> str:
        """プロンプトを渡し、生成テキストを返す。

        Args:
            prompt: モデルに渡す完全なプロンプト文字列。
                    システムプロンプト・ツール一覧・履歴を組み立て済みで渡す前提。
            stop:   モデル生成を打ち切る文字列のリスト（任意）。
                    ReAct ループでは ``["Observation:"]`` を指定し、
                    モデルが Observation を捏造するのを防ぐ。

        Returns:
            モデルが生成した素のテキスト。後段のパースは呼び出し側の責任。
        """
```

### 契約の不変条件

- **同期 1 回応答**：ストリーミングは Phase 1 では返さない。
- **副作用なし**：このメソッドは I/O 以外の状態を変えない。トークン履歴の管理は呼び出し側。
- **`stop` の扱い**：実装は `stop` 文字列の**直前**で生成を打ち切る。`stop` 文字列自体は出力に含めない（含めても呼び出し側が削る前提でよいが、含めない実装を推奨）。
- **失敗時**：HTTP エラー・タイムアウト等は例外を送出して呼び出し側に伝える（黙って空文字列を返さない）。

### 採用理由（要約）

- `generate(prompt, *, stop=None)` を採用。代替案 `chat(prompt) -> str` は捨てた。
- 主な理由：**ReAct ループでは `stop=["Observation:"]` が必須**。これがないと小型ローカルモデルが Observation を捏造して破綻する（design.md §4.1 / §7 で言及）。
- 詳細とトレードオフは [`tradeoffs.md`](tradeoffs.md) を参照。

---

## 2. `Tool` — ツール抽象

### 構造

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class Tool:
    name: str
    """ツール名。例: "download_file"。LLM が Action: で参照する識別子。"""

    description: str
    """LLM がいつ使うべきか判断するための説明。プロンプトに直接埋め込まれる。"""

    parameters: dict
    """JSON Schema 形式の引数定義。プロンプトにも表示される。"""

    func: Callable[..., str]
    """実行関数。キーワード引数で arguments を受け、結果は文字列で返す。
       戻り文字列はそのまま Observation として LLM に差し戻される。"""
```

### 契約の不変条件

- **戻り値は必ず `str`**：LLM への差し戻し（Observation）に直結するため。dict や object を返さない。
- **例外は呼び出し側で吸収**：`Tool.func` 自体は例外を投げてよい。`ToolRegistry.invoke` がそれを捕捉して失敗内容を Observation 文字列に変換する（後述）。
- **冪等性は要求しない**：ファイル DL や移動は非冪等で構わない。ただし副作用は明示的であること（隠れた destructive 動作禁止）。

---

## 3. `ToolRegistry` — ツール束の登録/検索/実行

### 署名

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None:
        """ツールを登録する。同名がある場合は例外。"""

    def get(self, name: str) -> Tool:
        """名前でツールを取得。未登録の場合は KeyError。"""

    def render_for_prompt(self) -> str:
        """登録済みツールをプロンプト埋め込み用テキストに整形。
           name / description / parameters(JSON Schema) を含む。"""

    def invoke(self, name: str, arguments: dict) -> str:
        """ツールを実行し Observation 文字列を返す。
           未登録ツール名・実行例外は **どちらも Observation 文字列に変換**する
           （LLM がリトライ・別手段を選べるようにするため、致命扱いにしない）。"""
```

### 契約の不変条件

- **`invoke` は決して例外を投げない**：失敗内容を Observation 文字列に変換して返す。ReAct ループを止めないため。
- **`render_for_prompt` の出力フォーマット**：Phase 1 では各ツールにつき以下の形：

  ```
  - <name>: <description>
    parameters: <JSON Schema を1行 JSON で>
  ```

  改行・順序の差異は LLM 側に影響するため、変更時は両者合意必須。

---

## 4. ReAct 出力フォーマット（モデル側に要求する形式）

### モデルが出力すべきフォーマット

```
Thought: <次に何をすべきかの思考>
Action: <ツール名>
Action Input: <JSON 形式の引数>
```

ツール不要時は：

```
Thought: <最終的な判断>
Final Answer: <ユーザへの回答>
```

### パーサ側の契約

- **行頭ラベル**（`Thought:` / `Action:` / `Action Input:` / `Final Answer:`）で分割する。
- **`Action Input` は JSON としてパース**。失敗時は致命扱いせず、Observation で「JSON が不正」と返して LLM に再試行させる。
- **`stop=["Observation:"]`** を LLM に渡し、モデルが Observation を自己捏造しないようにする。
- **未知ツール名**は致命扱いせず、Observation で「未知のツールです。利用可能ツール: ...」と返す。
- **`max_steps`**：呼び出し側が指定（既定 10）。到達時は最後の出力をそのまま返すか、エラーとして応答するかを Agent 側で決める（Phase 1 は前者で可）。

### 凡例

- `Thought` は省略可とはしない（プロンプトで強制）。省略時はパーサ側でログ出力のみで進行可。
- 同一ターンで `Action` と `Final Answer` の両方が出た場合は **`Final Answer` を優先**して終了する。

---

## 5. 変更ルール

- このファイルは Phase 1 の**契約面**を固定するもの。
- 変更したい場合は両者（Sho / Yuki）の合意を取り、本ファイルと `tradeoffs.md` の該当エントリを同時更新する。
- 実装が契約に追従できなくなった場合は、**契約を直してから実装を直す**順序を守る（実装の都合で契約をなし崩しにしない）。
