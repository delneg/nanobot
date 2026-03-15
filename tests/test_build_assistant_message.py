from nanobot.utils.helpers import build_assistant_message


def test_build_assistant_message_basic() -> None:
    msg = build_assistant_message("hello")
    assert msg == {"role": "assistant", "content": "hello"}


def test_build_assistant_message_with_reasoning_details() -> None:
    details = [{"type": "thinking", "thinking": "let me think..."}]
    msg = build_assistant_message("answer", reasoning_details=details)
    assert msg["reasoning_details"] == details
    assert msg["content"] == "answer"


def test_build_assistant_message_reasoning_details_none_omitted() -> None:
    msg = build_assistant_message("hi", reasoning_details=None)
    assert "reasoning_details" not in msg


def test_build_assistant_message_all_fields() -> None:
    tools = [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    details = [{"type": "thinking", "thinking": "hmm"}]
    blocks = [{"type": "thinking", "thinking": "block"}]
    msg = build_assistant_message(
        "content",
        tool_calls=tools,
        reasoning_content="reason",
        reasoning_details=details,
        thinking_blocks=blocks,
    )
    assert msg["tool_calls"] == tools
    assert msg["reasoning_content"] == "reason"
    assert msg["reasoning_details"] == details
    assert msg["thinking_blocks"] == blocks
