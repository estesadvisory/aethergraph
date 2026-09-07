from __future__ import annotations

import math
import re

from aethergraph.types import Modality, TaskClassification, TaskType

_CODE_HINTS = (
    r"\b(def|class|function|import|package|fn |impl |interface|typedef)\b",
    r"```",
    r"\b(refactor|compile|unit test|stacktrace|typescript|python|golang|rust)\b",
    r"\b(bug|fix this|implement|code review|pull request)\b",
)
_REASON_HINTS = (
    r"\b(prove|derive|theorem|why does|tradeoff|compare|analyze|root cause)\b",
    r"\b(step by step|first principles|constraint|optimize for)\b",
    r"[=<>]{1,2}.+\d",
)
_EXTRACT_HINTS = (
    r"\b(extract|parse|json|schema|fields?|entities|structured)\b",
    r"\{.+:.+\}",
)
_CLASSIFY_HINTS = (
    r"\b(classify|label|categorize|sentiment|spam|priority)\b",
)
_SUMMARIZE_HINTS = (
    r"\b(summarize|tldr|digest|executive summary|key points)\b",
)
_WRITE_HINTS = (
    r"\b(write|draft|rewrite|blog|essay|press release|email)\b",
)
_TRANSLATE_HINTS = (
    r"\b(translate|translation|into (spanish|french|german|japanese|chinese))\b",
)
_PLAN_HINTS = (
    r"\b(plan|decompose|roadmap|work breakdown|milestones|steps to)\b",
)
_VISION_HINTS = (
    r"\b(image|screenshot|photo|diagram|ui mock|picture)\b",
)
_EMBED_HINTS = (
    r"\b(embed|embedding|vector|similarity search)\b",
)

_PATTERNS: list[tuple[TaskType, tuple[str, ...], float]] = [
    (TaskType.CODE, _CODE_HINTS, 1.4),
    (TaskType.REASONING, _REASON_HINTS, 1.3),
    (TaskType.EXTRACTION, _EXTRACT_HINTS, 1.5),
    (TaskType.CLASSIFICATION, _CLASSIFY_HINTS, 1.6),
    (TaskType.SUMMARIZATION, _SUMMARIZE_HINTS, 1.5),
    (TaskType.WRITING, _WRITE_HINTS, 1.1),
    (TaskType.TRANSLATION, _TRANSLATE_HINTS, 1.7),
    (TaskType.PLANNING, _PLAN_HINTS, 1.2),
    (TaskType.VISION, _VISION_HINTS, 1.8),
    (TaskType.EMBEDDING, _EMBED_HINTS, 2.0),
]


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _score_patterns(text: str) -> dict[TaskType, float]:
    lowered = text.lower()
    scores = {task: 0.0 for task in TaskType}
    scores[TaskType.CONVERSATION] = 0.4
    for task_type, patterns, weight in _PATTERNS:
        hits = 0.0
        for pattern in patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
                hits += 1
        scores[task_type] += hits * weight
    if re.search(r"```[a-zA-Z]*\n", text):
        scores[TaskType.CODE] += 1.5
    if len(text) > 4000 and scores[TaskType.SUMMARIZATION] == 0:
        scores[TaskType.SUMMARIZATION] += 0.3
    return scores


def _complexity(text: str, task_type: TaskType, score: float) -> int:
    length = len(text)
    complexity = 1
    if length > 400:
        complexity += 1
    if length > 2000:
        complexity += 1
    if score >= 3:
        complexity += 1
    if re.search(r"\b(must|exactly|strict|production|prove|multi-file)\b", text, re.I):
        complexity += 1
    if task_type in {TaskType.REASONING, TaskType.CODE, TaskType.PLANNING} and length > 800:
        complexity += 1
    return max(1, min(5, complexity))


def classify(prompt: str, modalities: list[Modality] | None = None) -> TaskClassification:
    modalities = modalities or [Modality.TEXT]
    scores = _score_patterns(prompt)
    if Modality.IMAGE in modalities:
        scores[TaskType.VISION] += 3
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    task_type, top_score = ranked[0]
    if top_score < 0.8:
        task_type = TaskType.CONVERSATION
    complexity = _complexity(prompt, task_type, top_score)
    rationale = ", ".join(f"{name.value}={value:.1f}" for name, value in ranked[:4])
    return TaskClassification(
        task_type=task_type,
        complexity=complexity,
        modalities=modalities,
        scores={key.value: value for key, value in scores.items()},
        rationale=rationale,
    )
