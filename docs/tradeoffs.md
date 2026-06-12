# CreateAI 技術的トレードオフログ（Yuki）

> ランタイム基盤の設計判断と、それを選んだ理由・代替案・捨てたものを記録する。
> 「コードを書いた理由」を後から再構成できる状態を維持することが目的。
> Sho の意思決定ログは [`decisions.md`](decisions.md) を参照。

---

## エントリ #1 — `LLMClient` の署名: `generate(prompt, *, stop=None) -> str`

**日付**: 2026-06-12
**選択**: `LLMClient.generate(prompt: str, *, stop: list[str] | None = None) -> str`
**捨てた代替**: `LLMClient.chat(prompt: str) -> str`

### 何が論点だったか

計画書（スケジュール文書）では `chat(prompt) -> str` を Week0 で固定する案だったが、設計書 (`docs/design.md` §4.1) では `generate(prompt, *, stop=None)` を提案していた。Week0 ゲートで両者の不一致が露見した。

### なぜ `generate` 側を選んだか

最終的に `stop` パラメータの有無が決め手になった。

- **ReAct ループでは `stop=["Observation:"]` が必須**。これがないと小型ローカルモデルは Observation を**自分で続けて書いてしまう**。ツール結果を「捏造」するということ。`design.md` §4.1 と §7 の双方でこの点が明示されている。
- 計画書側の `chat()` シグネチャだと、後から `stop` を追加するために契約変更 → 全実装の書き直し、というデグレを Week2 で踏むリスクが高い。
- Week2 が**最重要ボトルネック週**（ReActフォーマット遵守のトライ&エラー集中週）であることを踏まえると、`stop` を最初から契約に入れておく方が後で得をする。

### 何を捨てたか

- **API のシンプルさ**: `chat()` の方が呼び出し側が短く書ける。ただし、Phase 1 のスコープでは呼び出し箇所はほぼ `Agent` 内の1箇所のみで、ここでの簡潔性ゲインは小さい。
- **OpenAI チャット風の親和性**: 業界の `chat.completions` 風 API への馴染みやすさを捨てた。ただしどちらに転んでも実装が変わるだけで、Phase 1 のスコープでは差し障りなし。

### 検証のしるし

Week2 でモデルが `Observation:` を続けて生成しようとする現象が出たとき、`stop=["Observation:"]` 1行で止まることを確認する。止まらなければこの設計判断は破綻しているので、本エントリを更新する。

---

## エントリ #2 — `LLMClient` 抽象の厚さ: 薄い Protocol + 単一メソッド

**日付**: 2026-06-12
**選択**: `typing.Protocol` で 1 メソッド (`generate`) のみ定義
**捨てた代替**: (a) リッチクライアント（メッセージ履歴管理 / トークン数推定 / リトライ内蔵）、(b) `langchain` 互換の `BaseChatModel` 派生

### なぜ薄い Protocol を選んだか

- `design.md` §9 の**最小依存方針**と一致する。`langchain` 等の重厚フレームワークに依存すると、Phase 1 のゴール（自前 ReAct ループで挙動を完全制御）と衝突する。
- 履歴管理・トークン推定・リトライは**呼び出し側（Agent）が責任を持つ**方が、ReAct ループの制御を `Agent` 内に閉じ込められる。`LLMClient` に分散させると、Week2 のフォーマット遵守問題のデバッグ箇所が広がる。
- `Protocol` (構造的部分型) を使うことで、差し替え時の継承木の制約を回避できる。将来 `LMStudioClient` / `LlamaCppClient` / `MockLLMClient` を作るときに、ベースクラス継承を強制せずに済む。

### 何を捨てたか

- **メッセージ履歴の自動管理**: `LLMClient` が会話履歴を持たない設計。Phase 2 以降でストリーミングや multi-turn API を導入するなら、別 Protocol を追加して棲み分ける予定。
- **リトライ内蔵**: HTTP エラーは raise する。リトライ戦略は呼び出し側の責任。Phase 1 では Ollama がローカルにいる前提でネットワーク不安定性は無視できる。
- **`langchain` 互換**: 互換性のための余計な abstraction (Message / Tool 二重定義 / Callbacks) を全て捨てた。

### 検証のしるし

`OllamaClient` の置き換え用に `MockLLMClient` を書く際、`generate(prompt, stop=None)` だけ実装すれば差し込めることを確認する（Week2 で Sho の place_file 統合に使う想定）。継承や追加メソッドが必要になった時点でこの判断は限界。

---

*次エントリ予定（Week2 以降）*:
- ReAct ループ構造の選択（自前小ループ vs ステートマシン）
- 出力パーサのアプローチ（正規表現 / 行ベース / 構造化出力モード）
- file 系ツール抽象の設計判断（Week3）
