import os
import sys

from deepeval.evaluate import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
from agent_instrumented import support_agent
from llm_config import build_deepeval_model

def actual_output(user_input: str) -> str:
    return support_agent(user_input)

# user_input1 = "Where is my order ORD-1042?",
# test_case1 = LLMTestCase(
#     input = "Where is my order ORD-1042?",
#     actual_output = actual_output(input)
# )

def test_task_completion_refund_policy():
    user_input2 = "What is the refund policy for electronics?"
    test_case2 = LLMTestCase(
        input = user_input2,
        actual_output = actual_output(user_input2)
    )

    evaluate(test_cases = [test_case2],
             metrics= [TaskCompletionMetric(threshold=0.7,model= build_deepeval_model())])
