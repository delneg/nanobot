from nanobot.agent.context import ContextBuilder
from nanobot.agent.loop import AgentLoop
from nanobot.session.manager import Session


def _mk_loop() -> AgentLoop:
    loop = AgentLoop.__new__(AgentLoop)
    loop._TOOL_RESULT_MAX_CHARS = 500
    return loop


def test_save_turn_skips_multimodal_user_when_only_runtime_context() -> None:
    loop = _mk_loop()
    session = Session(key="test:runtime-only")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [{"role": "user", "content": [{"type": "text", "text": runtime}]}],
        skip=0,
    )
    assert session.messages == []


def test_save_turn_keeps_image_placeholder_after_runtime_strip() -> None:
    loop = _mk_loop()
    session = Session(key="test:image")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": runtime},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }],
        skip=0,
    )
    assert session.messages[0]["content"] == [{"type": "text", "text": "[image]"}]


def test_save_turn_strips_video_base64_to_placeholder() -> None:
    loop = _mk_loop()
    session = Session(key="test:video")
    runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now (UTC)"

    loop._save_turn(
        session,
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": runtime},
                {"type": "image_url", "image_url": {"url": "data:video/mp4;base64,AAAA"}},
                {"type": "text", "text": "check this video"},
            ],
        }],
        skip=0,
    )
    saved = session.messages[0]["content"]
    assert saved == [
        {"type": "text", "text": "[video]"},
        {"type": "text", "text": "check this video"},
    ]


def test_save_turn_strips_mixed_image_and_video() -> None:
    loop = _mk_loop()
    session = Session(key="test:mixed-media")

    loop._save_turn(
        session,
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/"}},
                {"type": "image_url", "image_url": {"url": "data:video/mp4;base64,AAAA"}},
                {"type": "text", "text": "describe both"},
            ],
        }],
        skip=0,
    )
    saved = session.messages[0]["content"]
    assert saved == [
        {"type": "text", "text": "[image]"},
        {"type": "text", "text": "[video]"},
        {"type": "text", "text": "describe both"},
    ]
