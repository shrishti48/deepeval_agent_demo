# Import os so the test can build paths relative to this file.
import os
# Import sys so the repo root can be inserted into Python's import path.
import sys
#
# Import pytest for parametrized test execution over the dataset goldens.
import pytest
#
# Import DeepEval's assert helper to score each golden after the app runs.
from deepeval import assert_test
# Import the active golden accessor so expected output can be copied onto the trace.
from deepeval.contextvars import get_current_golden
# Import dataset types used to load the goldens file and type the test parameter.
from deepeval.dataset import EvaluationDataset, Golden
# Import tracing helpers to create a top-level traced wrapper around the app call.
from deepeval.tracing import observe, update_current_trace
#
# Add the repository root to sys.path so local imports resolve when the test runs directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# Import the shared RAG metric list defined for this evaluation suite.
from evals.metrics import RAG_TRACE_METRICS
# Import the real RAG QA bot under a private alias so the traced wrapper stays explicit.
from app.rag_agent import rag_qa_bot as _rag_qa_bot
#
#
# Mark this wrapper as an observed DeepEval trace entry point named rag_qa_bot.
@observe(name="rag_qa_bot")
# Define a thin traced wrapper that runs the real bot and enriches the trace first.
def run_traced_rag_qa_bot(user_input: str) -> str:
    # Read the current golden from DeepEval's context so this run can use its reference output.
    golden = get_current_golden()
    # Only attach expected_output when the golden exists and actually defines one.
    if golden and golden.expected_output:
        # Copy the expected output onto the current trace so trace-based metrics can read it.
        update_current_trace(expected_output=golden.expected_output)
    # Run the real RAG bot and return its answer unchanged.
    return _rag_qa_bot(user_input)
#
#
# Create the dataset container that will load the committed goldens file.
dataset = EvaluationDataset()
# Populate the dataset from the JSON file that sits next to this test module.
dataset.add_goldens_from_json_file(
    # Build the real goldens file path from the repo's data directory.
    file_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "knowledge_base",
        "goldens.json",
    )
)
#
#
# Parametrize the test so pytest runs one evaluation per golden in the dataset.
@pytest.mark.parametrize("golden", dataset.goldens)
# Define the single end-to-end RAG evaluation test for one golden.
def test_rag_qa_bot(golden: Golden):
    # Execute the traced RAG bot with the golden input so DeepEval records the run.
    run_traced_rag_qa_bot(golden.input)
    # Assert the recorded trace against the configured RAG metrics for this suite.
    assert_test(golden=golden, metrics=RAG_TRACE_METRICS)
