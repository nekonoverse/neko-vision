# neko-vision

llama.cpp 連携画像自動タグ付け/キャプション生成マイクロサービス。

[Nekonoverse](https://github.com/nekonoverse/nekonoverse) のアップロード画像に対して、llama.cpp の OpenAI 互換 API 経由でタグとキャプションを自動生成する。

## デフォルトモデル

**Gemma 4 E2B** (Per-Layer Embeddings, 実効 2B パラメータ)

- Vision/Text マルチモーダル対応
- Q4_K_M 量子化で約 3.1 GB
- コンテキスト長 128K 対応（本サービスでは 16K に設定）
- Audio 入力にもネイティブ対応（本サービスでは未使用）

## セットアップ

### 1. モデルファイルの配置

`./models` ディレクトリに GGUF ファイルをダウンロードする。ホストに Python や追加ツールをインストールせず、Docker コンテナ経由で取得できる。

```bash
mkdir -p models

docker run --rm -v ./models:/models python:3.12-slim bash -c \
  "pip install -q huggingface-hub && \
   hf download unsloth/gemma-4-E2B-it-GGUF \
     gemma-4-E2B-it-Q4_K_M.gguf \
     mmproj-BF16.gguf \
     --local-dir /models"
```

> `curl` では Hugging Face の LFS リダイレクトで HTML が保存される場合がある。`huggingface-cli` を使うこと。

> **注意**: ファイル名はリポジトリにより異なる場合がある。`docker-compose.yml` の `command` 内のファイル名と実際に配置したファイル名を一致させること。

### 2. 起動

```bash
cp docker-compose.yml.example docker-compose.yml
docker compose up -d
```

### 3. Nekonoverse 本体との接続

Nekonoverse の環境変数に以下を設定:

```
NEKO_VISION_URL=http://<neko-vision-host>:8004/tag
```

## 環境変数

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `LLAMA_URL` | `http://localhost:8080` | llama.cpp server の URL |

## モデルの変更

他のサイズの Gemma 4 モデルに変更する場合は、`docker-compose.yml` の `command` 内の `-m` と `--mmproj` のパスを差し替える。

```yaml
command: >
  -m /models/<model-file>.gguf
  --mmproj /models/<mmproj-file>.gguf
  --host 0.0.0.0
  --port 8080
  -c 16384
  -np 2
```

利用可能なモデルサイズ:
- **E2B** (2B) — 軽量、エッジ向け（デフォルト）
- **E4B** (4B) — バランス型
- **26B-A4B** (26B, active 4B) — MoE、高精度
- **31B** (31B) — 最高精度、要大容量 VRAM

GPU を使う場合は llama イメージを `:server-cuda` に変更し、`nvidia-container-toolkit` をインストールすること。

## Ollama からの移行

以前のバージョンでは Ollama (`gemma3:4b`) を使用していたが、llama.cpp server に移行した。

- `OLLAMA_URL` / `OLLAMA_MODEL` 環境変数は **削除済み**。`LLAMA_URL` を使用すること
- デフォルトモデルが `gemma3:4b` → **Gemma 4 E2B** に変更
- API 契約 (`POST /tag`, `GET /health`) は互換性を維持

## API

### `POST /tag`

画像にタグとキャプションを付与する。

```json
// Request
{
  "image": "<base64エンコードされた画像>",
  "text": "ノート本文（省略可）",
  "context": ["親ノート1", "親ノート2"]
}

// Response
{
  "tags": ["猫", "かわいい", "写真"],
  "caption": "かわいい猫の写真"
}
```

### `GET /health`

ヘルスチェック。

```json
{
  "status": "ok",
  "backend": "llama.cpp",
  "llama_url": "http://llama:8080",
  "llama_connected": true
}
```
