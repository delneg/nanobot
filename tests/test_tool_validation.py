from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.loader import convert_keys, convert_to_camel


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


def _build_mcp_env_data() -> dict[str, Any]:
    return {
        "tools": {
            "mcpServers": {
                "demo": {
                    "command": "npx",
                    "env": {
                        "OPENAI_API_KEY": "test_key",
                        "MyCustomToken": "abc",
                    },
                }
            }
        }
    }


def test_convert_keys_preserves_mcp_env_var_names() -> None:
    data = _build_mcp_env_data()

    converted = convert_keys(data)
    env = converted["tools"]["mcp_servers"]["demo"]["env"]

    assert env["OPENAI_API_KEY"] == "test_key"
    assert env["MyCustomToken"] == "abc"


def test_convert_to_camel_preserves_mcp_env_var_names() -> None:
    data = {
        "tools": {
            "mcp_servers": {
                "demo": {
                    "command": "npx",
                    "env": {
                        "OPENAI_API_KEY": "test_key",
                        "MyCustomToken": "abc",
                    },
                }
            }
        }
    }

    converted = convert_to_camel(data)
    env = converted["tools"]["mcpServers"]["demo"]["env"]

    assert env["OPENAI_API_KEY"] == "test_key"
    assert env["MyCustomToken"] == "abc"


def test_convert_keys_still_converts_non_env_keys() -> None:
    data = {
        "tools": {
            "restrictToWorkspace": True,
            "mcpServers": {
                "demo": {
                    "extraHeaders": {"XCustom": "v"},
                }
            },
        }
    }

    converted = convert_keys(data)
    tools = converted["tools"]

    assert "restrict_to_workspace" in tools
    assert "mcp_servers" in tools
    assert "extra_headers" in tools["mcp_servers"]["demo"]
