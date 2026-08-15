import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.semantic_similarity_metric import SemanticSimilarityMetric

from deepeval.test_case import LLMTestCase

def test_reference_based_semantic_similarity():
    metric = SemanticSimilarityMetric(threshold=0.75)

    test_case = LLMTestCase(
        input="What's your return window?",
        actual_output="You're eligible for a 30-day full refund at no extra cost.",
        expected_output="30-day full refund at no extra cost.",
    )

    score = metric.measure(test_case)

    assert isinstance(score, float)
