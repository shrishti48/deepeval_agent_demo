import os
import sys

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_config import build_deepeval_model


def test_rag_like_metrics():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="All customers are eligible for a 30 day full refund at no extra costs.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."],
        context=["All customers are eligible for a 30 day full refund at no extra costs."],
    )

    judge_model = build_deepeval_model()

    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model=judge_model,
    )

    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model=judge_model,
    )

    hallucination = HallucinationMetric(
        threshold=0.5,   # maximum allowed score
        model=judge_model,
    )

    evaluate(
        [test_case],
        [answer_relevancy, faithfulness, hallucination],
    )
