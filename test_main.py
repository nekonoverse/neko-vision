"""neko-vision ユニットテスト。Ollama をモックしてテストする。"""

import base64
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import Response

from main import (
    _build_prompt,
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
    assert "model" in data
    assert "ollama_url" in data


def _make_image_base64() -> str:
    """テスト用の最小限のbase64画像データを生成する。"""
    return base64.b64encode(b"\x00" * 200).decode()


def _make_ollama_response(status_code: int, json_body: dict) -> Response:
    """raise_for_status() が動作するようにリクエスト付きのレスポンスを生成する。"""
    request = httpx.Request("POST", "http://localhost:11434/api/generate")
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
    ollama_response = {
        "response": '{"tags": ["猫", "かわいい"], "caption": "かわいい猫の写真"}'
    }
    mock_resp = _make_ollama_response(200, ollama_response)

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
    ollama_response = {
        "response": '{"tags": ["猫", "うちの子"], "caption": "飼い猫の写真"}'
    }
    mock_resp = _make_ollama_response(200, ollama_response)

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
    # Ollama へのリクエストにテキストが含まれることを確認
    call_args = instance.post.call_args
    assert "うちの猫です" in call_args.kwargs["json"]["prompt"]


@pytest.mark.asyncio
async def test_tag_ollama_failure(client):
    with patch("main.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.post.side_effect = httpx.ConnectError("connection refused")
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = instance

        resp = await client.post("/tag", json={"image": _make_image_base64()})

    assert resp.status_code == 502
