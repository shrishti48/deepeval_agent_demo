from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)

from llm_config import build_deepeval_model


RAG_TRACE_METRICS = [
    AnswerRelevancyMetric(
        threshold=0.7,
        model=build_deepeval_model(),
        include_reason=True,
    ),
    FaithfulnessMetric(
        threshold=0.7,
        model=build_deepeval_model(),
        include_reason=True,
    ),
    ContextualPrecisionMetric(
        threshold=0.7,
        model=build_deepeval_model(),
        include_reason=True,
    ),
    ContextualRecallMetric(
        threshold=0.7,
        model=build_deepeval_model(),
        include_reason=True,
    ),
]

AnswerRelevancyMetric = [AnswerRelevancyMetric(
        threshold=0.7,
        model=build_deepeval_model(),
        include_reason=True,
    )]

