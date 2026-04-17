"""neko-vision: llama.cpp連携画像自動タグ付け/キャプション生成マイクロサービス。

llama.cpp (gemma-4 等) を利用してアップロード画像にタグとキャプションを付与する。
ノート本文テキストやリプライツリーをコンテキストとして渡すことで、
固有名詞の捕捉精度が向上する。
"""

import base64
import json
import logging
import os
import re
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("neko-vision")
logging.basicConfig(level=logging.INFO)

LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8080")
MAX_RETRIES = 2

# プロンプトテンプレート
_PROMPT_IMAGE_ONLY = (
    "画像を分析し、以下のJSON形式のみ出力してください。\n"
    "tagsは画像の内容を表す日本語の単語を3〜7個、\n"
    "captionは画像を説明する日本語の短い文を1つ。\n"
    '{"tags": ["タグ1", "タグ2", ...], "caption": "説明文"}'
)

_PROMPT_WITH_TEXT = (
    "画像とそれに添付された投稿文を分析し、以下のJSON形式のみ出力してください。\n"
    "tagsは画像+投稿文の内容を表す日本語の単語を3〜7個、\n"
    "captionは画像を説明する日本語の短い文を1つ。\n"
    "投稿文も参考にしてより的確なタグとキャプションを生成してください。\n"
    '{{"tags": ["タグ1", "タグ2", ...], "caption": "説明文"}}\n\n'
    "投稿文: {text}"
)

_PROMPT_WITH_CONTEXT = (
    "画像と投稿文、会話の流れを分析し、以下のJSON形式のみ出力してください。\n"
    "tagsは画像+投稿文+文脈の内容を表す日本語の単語を3〜7個、\n"
    "captionは画像を説明する日本語の短い文を1つ。\n"
    "会話の流れと投稿文も参考にしてより的確なタグとキャプションを生成してください。\n"
    '{{"tags": ["タグ1", "タグ2", ...], "caption": "説明文"}}\n\n'
    "会話の流れ:\n{context}\n\n"
    "投稿文: {text}"
)

_llama_ok: bool = False


def _detect_mime(image_bytes: bytes) -> str:
    """画像バイト列の magic bytes から MIME タイプを判定する。"""
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return "image/jpeg"


def _build_prompt(text: str | None, context: list[str] | None) -> str:
    """テキストとコンテキストの有無に応じてプロンプトを組み立てる。"""
    if context and text:
        ctx_text = "\n".join(f"- {c}" for c in context[-5:])
        return _PROMPT_WITH_CONTEXT.format(text=text[:500], context=ctx_text)
    if text:
        return _PROMPT_WITH_TEXT.format(text=text[:500])
    return _PROMPT_IMAGE_ONLY


def _parse_json_response(text: str) -> dict | None:
    """レスポンスからJSONを抽出する。マークダウンフェンス対応。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        cleaned = "\n".join(lines[start:end])

    # JSON部分を抽出（前後に余分なテキストがある場合）
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _validate_result(parsed: dict) -> dict:
    """パース結果をバリデーションし正規化する。"""
    tags = parsed.get("tags", [])
    caption = parsed.get("caption", "")

    if not isinstance(tags, list):
        tags = []
    tags = [str(t) for t in tags if t][:15]

    if not isinstance(caption, str):
        caption = ""
    caption = caption[:500]

    return {"tags": tags, "caption": caption}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時にllama.cppへの接続を確認する。"""
    global _llama_ok
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{LLAMA_URL}/health")
            if resp.status_code == 200:
                _llama_ok = True
                logger.info("llama.cpp 接続OK: %s", LLAMA_URL)
            else:
                logger.warning("llama.cpp 接続失敗: status %d", resp.status_code)
    except Exception as e:
        logger.warning("llama.cpp 接続失敗: %s", e)
    yield


app = FastAPI(title="neko-vision", lifespan=lifespan)


class TagRequest(BaseModel):
    image: str = Field(description="Base64エンコードされた画像データ")
    text: str | None = Field(default=None, description="ノート本文テキスト")
    context: list[str] | None = Field(
        default=None, description="リプライツリーの親ノート本文（古い順）"
    )


class TagResponse(BaseModel):
    tags: list[str] = Field(description="画像の内容を表す日本語タグ")
    caption: str = Field(description="画像を説明する日本語キャプション")


@app.post("/tag", response_model=TagResponse)
async def tag_image(request: TagRequest):
    """画像にタグとキャプションを付与する。"""
    # base64 バリデーション
    try:
        image_data = base64.b64decode(request.image)
        if len(image_data) < 100:
            raise ValueError("too small")
    except Exception:
        raise HTTPException(status_code=400, detail="不正なbase64画像データ")

    prompt = _build_prompt(request.text, request.context)
    mime = _detect_mime(image_data)
    image_url = f"data:{mime};base64,{request.image}"
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{LLAMA_URL}/v1/chat/completions",
                    json={
                        "model": "gemma4",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_url},
                                    },
                                ],
                            }
                        ],
                        "response_format": {"type": "json_object"},
                        # Gemma のデフォルト temperature=1.0 だと JSON 構造が崩れやすいため
                        # 安定したタグ出力を得るために低めに設定
                        "temperature": 0.2,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            last_error = str(e)
            logger.warning("llama.cpp 呼び出し失敗 (試行 %d): %s", attempt + 1, e)
            continue

        response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _parse_json_response(response_text)

        if parsed and ("tags" in parsed or "caption" in parsed):
            result = _validate_result(parsed)
            logger.info(
                "タグ付け成功: tags=%d, caption=%d文字",
                len(result["tags"]),
                len(result["caption"]),
            )
            return TagResponse(**result)

        last_error = f"JSONパース失敗: {response_text[:100]}"
        logger.warning("JSONパース失敗 (試行 %d): %s", attempt + 1, response_text[:100])

    raise HTTPException(
        status_code=502,
        detail=f"タグ付け失敗 ({MAX_RETRIES + 1}回試行): {last_error}",
    )


@app.get("/health")
async def health():
    """ヘルスチェック。"""
    return {
        "status": "ok",
        "backend": "llama.cpp",
        "llama_url": LLAMA_URL,
        "llama_connected": _llama_ok,
    }
