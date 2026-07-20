import json

import pytest

from bot import (
    Turn,
    UserSettings,
    actual_reply,
    correction_markup,
    parse_turn,
    settings_markup,
    system_prompt,
    topic_markup,
    trim_trailing_pcm16,
)
from topics import TOPICS


def test_parse_turn() -> None:
    payload = {"heard": "Ich gehen.", "corrected": "Ich gehe.", "reply": "Wohin gehst du?"}
    assert parse_turn(json.dumps(payload)) == Turn(**payload)


def test_parse_turn_accepts_markdown_fence() -> None:
    text = '```json\n{"heard":"Hallo","corrected":"Hallo","reply":"Wie geht es dir?"}\n```'
    assert parse_turn(text).reply == "Wie geht es dir?"


def test_parse_turn_rejects_missing_field() -> None:
    with pytest.raises(ValueError):
        parse_turn('{"heard":"Hallo","corrected":"Hallo"}')


def test_correction_markup_replaces_words_inline() -> None:
    result = correction_markup("Ich gehen heute nach Hause.", "Ich gehe heute nach Hause.")
    assert result == "Ich <s>gehen</s> <b>gehe</b> heute nach Hause."


def test_correction_markup_escapes_html() -> None:
    result = correction_markup("Ich mag <Kaffee>.", "Ich mag Kaffee.")
    assert "&lt;Kaffee&gt;" in result


def test_prompt_uses_selected_language_and_level() -> None:
    prompt = system_prompt(UserSettings(language="nb", level="B1"))
    assert "Norwegian Bokmål" in prompt
    assert "CEFR level B1" in prompt
    assert "do not translate" in prompt


def test_spanish_prompt_is_supported() -> None:
    prompt = system_prompt(UserSettings(language="es", level="A2"))
    assert "learning Spanish" in prompt
    assert "CEFR level A2" in prompt


def test_settings_keyboard_marks_current_values() -> None:
    keyboard = settings_markup(UserSettings(language="en", level="A1", show_reply_text=True))
    labels = [button.text for row in keyboard.inline_keyboard for button in row]
    assert "✓ English" in labels
    assert "✓ A1" in labels
    assert "📝 Текст ответа: включён" in labels


def test_audio_transcript_is_authoritative_reply() -> None:
    assert actual_reply("Planned question?", "Actually spoken answer.") == "Actually spoken answer."
    assert actual_reply("Fallback text.", "  ") == "Fallback text."


def test_topic_catalog_has_unique_scenarios() -> None:
    assert 20 <= len(TOPICS) <= 30
    assert len({topic.key for topic in TOPICS}) == len(TOPICS)


def test_active_topic_is_added_to_prompt() -> None:
    topic = TOPICS[0]
    prompt = system_prompt(UserSettings(language="de", level="A2"), topic)
    assert topic.situation in prompt
    assert topic.learner_goal in prompt
    assert topic.partner_role in prompt
    assert "Stay in character" in prompt


def test_topic_keyboard_has_new_and_stop_actions() -> None:
    callbacks = [button.callback_data for row in topic_markup().inline_keyboard for button in row]
    assert callbacks == ["topic:new", "topic:stop"]


def test_trailing_pcm_silence_is_trimmed_without_cutting_internal_audio() -> None:
    sample_rate = 24_000
    speech = (1000).to_bytes(2, "little", signed=True) * sample_rate
    silence = bytes(sample_rate * 5 * 2)
    result = trim_trailing_pcm16(speech + silence, sample_rate=sample_rate)
    assert len(speech) < len(result) <= len(speech) + sample_rate * 2
