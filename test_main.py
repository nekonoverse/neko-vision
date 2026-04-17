"""neko-vision ユニットテスト。llama.cpp をモックしてテストする。"""

import base64
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import Response

from main import (
    _build_prompt,
    _detect_mime,
    _ensure_jpeg_or_png,
    _parse_json_response,
    _validate_result,
    app,
)

# --- ヘルパー関数のテスト ---


def test_build_prompt_image_only():
    prompt = _build_prompt(None, None)
    assert "画像を分析し" in prompt
    assert "投稿文" not in prompt


def test_build_prompt_with_text():
    prompt = _build_prompt("猫の写真です", None)
    assert "投稿文" in prompt
    assert "猫の写真です" in prompt


def test_build_prompt_with_context():
    prompt = _build_prompt("返信です", ["親ノート1", "親ノート2"])
    assert "会話の流れ" in prompt
    assert "親ノート1" in prompt
    assert "返信です" in prompt


def test_build_prompt_text_truncation():
    long_text = "あ" * 1000
    prompt = _build_prompt(long_text, None)
    # テキストは500文字に切り詰められる
    assert "あ" * 500 in prompt
    assert "あ" * 501 not in prompt


def test_build_prompt_context_limit():
    context = [f"ノート{i}" for i in range(10)]
    prompt = _build_prompt("テスト", context)
    # コンテキストは最新5件のみ
    assert "ノート5" in prompt
    assert "ノート9" in prompt
    assert "ノート0" not in prompt


def test_parse_json_response_valid():
    result = _parse_json_response('{"tags": ["猫"], "caption": "猫の写真"}')
    assert result == {"tags": ["猫"], "caption": "猫の写真"}


def test_parse_json_response_markdown_fence():
    text = '```json\n{"tags": ["犬"], "caption": "犬"}\n```'
    result = _parse_json_response(text)
    assert result == {"tags": ["犬"], "caption": "犬"}


def test_parse_json_response_extra_text():
    text = 'はい、分析結果です。\n{"tags": ["風景"], "caption": "山"}\n以上です。'
    result = _parse_json_response(text)
    assert result["tags"] == ["風景"]


def test_parse_json_response_invalid():
    assert _parse_json_response("これはJSONではありません") is None


def test_parse_json_response_empty():
    assert _parse_json_response("") is None


def test_validate_result_normal():
    result = _validate_result({"tags": ["猫", "写真"], "caption": "猫の写真"})
    assert result == {"tags": ["猫", "写真"], "caption": "猫の写真"}


def test_validate_result_tags_limit():
    tags = [f"タグ{i}" for i in range(20)]
    result = _validate_result({"tags": tags, "caption": "テスト"})
    assert len(result["tags"]) == 15


def test_validate_result_invalid_types():
    result = _validate_result({"tags": "文字列", "caption": 123})
    assert result["tags"] == []
    assert result["caption"] == ""


def test_validate_result_empty_tags_filtered():
    result = _validate_result({"tags": ["猫", "", None, "犬"], "caption": "テスト"})
    assert result["tags"] == ["猫", "犬"]


# --- MIME 検出のテスト ---


def test_detect_mime_jpeg():
    assert _detect_mime(b"\xff\xd8\xff\xe0" + b"\x00" * 100) == "image/jpeg"


def test_detect_mime_png():
    assert _detect_mime(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) == "image/png"


def test_detect_mime_webp():
    data = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 100
    assert _detect_mime(data) == "image/webp"


def test_detect_mime_gif87a():
    assert _detect_mime(b"GIF87a" + b"\x00" * 100) == "image/gif"


def test_detect_mime_gif89a():
    assert _detect_mime(b"GIF89a" + b"\x00" * 100) == "image/gif"


def test_detect_mime_unknown():
    assert _detect_mime(b"\x00" * 200) == "image/jpeg"


# --- 画像変換のテスト ---


def test_ensure_jpeg_or_png_jpeg_passthrough():
    data = b"\xff\xd8\xff" + b"\x00" * 100
    result, mime = _ensure_jpeg_or_png(data, "image/jpeg")
    assert result is data
    assert mime == "image/jpeg"


def test_ensure_jpeg_or_png_png_passthrough():
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    result, mime = _ensure_jpeg_or_png(data, "image/png")
    assert result is data
    assert mime == "image/png"


def test_ensure_jpeg_or_png_webp_converts():
    """WebP を JPEG に変換できることを確認する。"""
    from io import BytesIO

    from PIL import Image

    # テスト用の小さい WebP を生成
    img = Image.new("RGB", (10, 10), color="red")
    buf = BytesIO()
    img.save(buf, format="WEBP")
    webp_data = buf.getvalue()

    result, mime = _ensure_jpeg_or_png(webp_data, "image/webp")
    assert mime == "image/jpeg"
    assert result != webp_data
    # 結果が有効な JPEG であることを確認
    assert result[:2] == b"\xff\xd8"


def test_ensure_jpeg_or_png_gif_converts():
    """GIF を JPEG に変換できることを確認する。"""
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (10, 10), color="blue")
    buf = BytesIO()
    img.save(buf, format="GIF")
    gif_data = buf.getvalue()

    result, mime = _ensure_jpeg_or_png(gif_data, "image/gif")
    assert mime == "image/jpeg"
    assert result[:2] == b"\xff\xd8"


# --- API エンドポイントのテスト ---


@pytest.fixture
def client():
    from httpx import ASGITransport

    transport = ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend"] == "llama.cpp"
    assert "llama_url" in data
    assert "llama_connected" in data


def _make_image_base64() -> str:
    """テスト用の最小限のbase64画像データを生成する。"""
    return base64.b64encode(b"\x00" * 200).decode()


def _make_llama_response(status_code: int, json_body: dict) -> Response:
    """raise_for_status() が動作するようにリクエスト付きのレスポンスを生成する。"""
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    return Response(status_code, json=json_body, request=request)


@pytest.mark.asyncio
async def test_tag_invalid_base64(client):
    resp = await client.post("/tag", json={"image": "invalid!!!"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_tag_too_small_image(client):
    tiny = base64.b64encode(b"\x00" * 10).decode()
    resp = await client.post("/tag", json={"image": tiny})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_tag_success(client):
    llama_response = {
        "choices": [
            {
                "message": {
                    "content": '{"tags": ["猫", "かわいい"], "caption": "かわいい猫の写真"}'
                }
            }
        ]
    }
    mock_resp = _make_llama_response(200, llama_response)

    with patch("main.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.post.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        resp = await client.post("/tag", json={"image": _make_image_base64()})

    assert resp.status_code == 200
    data = resp.json()
    assert data["tags"] == ["猫", "かわいい"]
    assert data["caption"] == "かわいい猫の写真"


@pytest.mark.asyncio
async def test_tag_with_text(client):
    llama_response = {
        "choices": [
            {
                "message": {
                    "content": '{"tags": ["猫", "うちの子"], "caption": "飼い猫の写真"}'
                }
            }
        ]
    }
    mock_resp = _make_llama_response(200, llama_response)

    with patch("main.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.post.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        resp = await client.post(
            "/tag",
            json={"image": _make_image_base64(), "text": "うちの猫です"},
        )

    assert resp.status_code == 200
    # llama.cpp へのリクエストにテキストが含まれることを確認
    call_args = instance.post.call_args
    messages = call_args.kwargs["json"]["messages"]
    # user メッセージの text content にプロンプト（テキスト含む）があることを確認
    text_content = messages[0]["content"][0]["text"]
    assert "うちの猫です" in text_content


@pytest.mark.asyncio
async def test_tag_400_returns_empty(client):
    """400 (画像デコード失敗等) はリトライせず空結果を返す。"""
    mock_resp = _make_llama_response(400, {"error": "failed to decode image bytes"})

    with patch("main.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.post.return_value = mock_resp
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        resp = await client.post("/tag", json={"image": _make_image_base64()})

    assert resp.status_code == 200
    data = resp.json()
    assert data["tags"] == []
    assert data["caption"] == ""
    # 400 ではリトライしないので1回だけ呼ばれる
    assert instance.post.call_count == 1


@pytest.mark.asyncio
async def test_tag_llama_failure(client):
    with patch("main.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.post.side_effect = httpx.ConnectError("connection refused")
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        resp = await client.post("/tag", json={"image": _make_image_base64()})

    assert resp.status_code == 502
