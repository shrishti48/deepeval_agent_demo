import os
import sys

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_config import build_deepeval_model

def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        evaluation_steps=[
            "Check whether the actual output contradicts anything stated in the expected output.",
            "Check whether the actual output is medically/factually reasonable given the input, even if it omits some detail present in the expected output.",
            "Do not penalize the actual output for being shorter or less detailed than the expected output, as long as nothing it says is wrong.",
            "Only fail the response if it contains an actual factual error or contradicts the expected output's core claims."
        ],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=build_deepeval_model(),
    )
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don't improve in a few days.",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
    )
    assert_test(test_case, [correctness_metric])
