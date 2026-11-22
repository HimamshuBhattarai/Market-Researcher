from pydantic import BaseModel

from market_research.services import llm


def test_strip_fences():
    assert llm._strip_fences("```json\n{\"ok\": true}\n```") == '{"ok": true}'


def test_generate_structured_uses_mocked_completion(monkeypatch):
    class SampleResponse(BaseModel):
        value: str

    monkeypatch.setattr(
        llm,
        "_complete_with_openai_compatible",
        lambda _messages: '{"value": "ok"}',
    )

    result = llm.generate_structured("Test", SampleResponse)
    assert result.value == "ok"
