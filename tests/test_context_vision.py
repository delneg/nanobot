"""Tests for ContextBuilder vision support and media handling."""

import pytest
from pathlib import Path

from nanobot.agent.context import ContextBuilder


class TestModelSupportsVision:
    """Tests for _model_supports_vision heuristic."""

    def test_none_model_assumes_vision(self) -> None:
        assert ContextBuilder._model_supports_vision(None) is True

    def test_known_vision_model_glm_4v(self) -> None:
        assert ContextBuilder._model_supports_vision("glm-4v") is True

    def test_known_vision_model_glm_46v(self) -> None:
        assert ContextBuilder._model_supports_vision("glm-4.6v") is True

    def test_known_vision_model_glm_46v_flash(self) -> None:
        assert ContextBuilder._model_supports_vision("glm-4.6v-flash") is True

    def test_known_vision_model_gpt4o(self) -> None:
        assert ContextBuilder._model_supports_vision("gpt-4o") is True

    def test_known_vision_model_qwen_vl(self) -> None:
        assert ContextBuilder._model_supports_vision("qwen-vl-max") is True

    def test_text_only_model_glm5_turbo(self) -> None:
        assert ContextBuilder._model_supports_vision("glm-5-turbo") is False

    def test_text_only_model_deepseek(self) -> None:
        # deepseek-chat has no vision pattern; litellm also returns False for unknown
        assert ContextBuilder._model_supports_vision("deepseek-chat") is False


class TestBuildUserContentVision:
    """Tests for _build_user_content with vision/non-vision models."""

    @pytest.fixture
    def builder(self, tmp_path: Path) -> ContextBuilder:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        return ContextBuilder(workspace)

    @pytest.fixture
    def image_file(self, tmp_path: Path) -> str:
        img = tmp_path / "test.jpg"
        # Write minimal JPEG header so detect_image_mime recognizes it
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        return str(img)

    def test_vision_model_returns_multipart(self, builder: ContextBuilder, image_file: str) -> None:
        result = builder._build_user_content("hello", [image_file], model="glm-4.6v")
        assert isinstance(result, list)
        types = [p["type"] for p in result]
        assert "image_url" in types
        assert "text" in types

    def test_non_vision_model_returns_text_with_refs(self, builder: ContextBuilder, image_file: str) -> None:
        result = builder._build_user_content("hello", [image_file], model="glm-5-turbo")
        assert isinstance(result, str)
        assert "hello" in result
        assert "[media (image/jpeg):" in result
        assert image_file in result

    def test_no_media_returns_text_regardless(self, builder: ContextBuilder) -> None:
        result = builder._build_user_content("hello", None, model="glm-5-turbo")
        assert result == "hello"

    def test_non_vision_missing_file_excluded(self, builder: ContextBuilder) -> None:
        result = builder._build_user_content("hello", ["/nonexistent/file.jpg"], model="glm-5-turbo")
        assert result == "hello"  # No refs added for missing files
