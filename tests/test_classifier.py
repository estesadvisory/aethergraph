from aethergraph.classifier import classify
from aethergraph.types import Modality, TaskType


def test_classifies_code():
    result = classify("Implement a FastAPI health endpoint and add a unit test.")
    assert result.task_type is TaskType.CODE
    assert result.complexity >= 1


def test_classifies_extraction():
    result = classify("Extract a JSON schema with fields name, cost, and provider from this catalog.")
    assert result.task_type is TaskType.EXTRACTION


def test_classifies_reasoning():
    result = classify("Prove why a local 8B model is cheaper than Sonnet for classification and analyze the tradeoff.")
    assert result.task_type is TaskType.REASONING


def test_vision_modality_overrides():
    result = classify("What is in this screenshot of a dashboard?", modalities=[Modality.TEXT, Modality.IMAGE])
    assert result.task_type is TaskType.VISION


def test_short_chat_defaults_to_conversation():
    result = classify("Hello there")
    assert result.task_type is TaskType.CONVERSATION
