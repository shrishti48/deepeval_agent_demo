import os
import sys

import pytest
import deepeval

from deepeval import assert_test

from deepeval.dataset import Golden, EvaluationDataset

from deepeval.test_case import LLMTestCase

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

from evals.golden_factories import BaseGoldenFactory, HappyPathGoldens, EdgeCaseGoldens, AdversarialGoldens, \
    MultiTopicGoldens

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.custom_eval_metrics import custom_faithfulness_metric, custom_relevance_metric, custom_coherence_metric, custom_tonality_metric
from app.qa_bot import qa_bot

# --- Step 1: define goldens (the "what I'm testing", written before running anything) ---

goldens = [

    # Easy, direct, single-fact lookups

    Golden(input="What if these shoes don't fit?", expected_output="You're eligible for a 30-day full refund at no extra cost."),

    Golden(input="How long does standard shipping take?", expected_output="Standard shipping takes 3-5 business days."),

    Golden(input="How do I reset my password?", expected_output="Click 'Forgot Password' on the login page and follow the emailed instructions."),

    Golden(input="Do gift cards expire?", expected_output="No, gift cards do not expire, but they can't be redeemed for cash."),

    # Medium: requires combining two facts from context

    Golden(input="If I return a defective item, when do I get my money back?", expected_output="Refunds are processed to your original payment method within 5-7 business days after the item is received."),

    Golden(input="I want faster delivery, what are my options?", expected_output="Express shipping is available for an extra fee, taking 1-2 business days."),

    # Edge case: information NOT in the knowledge base — tests whether the bot fabricates an answer

    Golden(input="Do you ship to Antarctica?", expected_output="The bot should say it doesn't have information on this, or that international shipping outside supported countries isn't available — it should NOT invent a specific policy."),

    Golden(input="What's your policy on price matching with competitors?", expected_output="The bot should say it doesn't have this information, since price matching isn't in the knowledge base at all."),

    # Edge case: ambiguous/underspecified question

    Golden(input="Can I cancel my order?", expected_output="Only within 1 hour of placing the order, before it enters processing."),

    # Edge case: locked account / security-adjacent question

    Golden(input="Why is my account locked?", expected_output="Accounts lock automatically after 5 failed login attempts, and unlock after 30 minutes."),

    # Slightly adversarial phrasing: same underlying question, worded very differently (tests paraphrase robustness from Day 3!)

    Golden(input="My package still hasn't shown up after a week, is that normal?", expected_output="Standard shipping takes 3-5 business days, so a week without delivery would be outside the normal window — the bot should flag this as longer than expected, not just restate the shipping policy blindly."),

    # Multi-part question

    Golden(input="What's your return window and how do refunds get paid back to me?", expected_output="30-day return window; refund goes to the original payment method within 5-7 business days of the item being received."),

]

# --- Step 2: build the dataset of goldens only; do not call the bot at import time ---

dataset = EvaluationDataset(goldens=goldens)

# --- Step 3: call the bot inside the parametrized test so requests happen during test execution ---

@pytest.mark.skip()
@pytest.mark.parametrize("golden", dataset.goldens)
def test_qa_bot_quality(golden: Golden):

    actual_output, retrieval_context = qa_bot(golden.input)

    test_case = LLMTestCase(

        input=golden.input,

        actual_output=actual_output,

        expected_output=golden.expected_output,

        retrieval_context=retrieval_context,

    )

    answer_relevancy = AnswerRelevancyMetric(threshold=0.6)

    faithfulness = FaithfulnessMetric(threshold=0.6)

    assert_test(test_case, [answer_relevancy, faithfulness])

@pytest.mark.skip()
@pytest.mark.parametrize("golden", dataset.goldens)
def test_qa_bot_full_evaluation(golden: Golden):
    actual_output, retrieval_context = qa_bot(golden.input)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        retrieval_context=retrieval_context,
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.6),
        FaithfulnessMetric(threshold=0.6),
        custom_faithfulness_metric,
        custom_relevance_metric,
        custom_coherence_metric,
        custom_tonality_metric
    ]

    assert_test(test_case, metrics)

def build_all_goldens(factories: list[BaseGoldenFactory]) -> list[Golden]:
    # this function only knows about the ABSTRACT BaseGoldenFactory interface —
    # it has no idea, and doesn't need to know, which concrete subclasses it's calling
    goldens = []
    for factory in factories:
        goldens.extend(factory.build_goldens())
    return goldens

all_goldens = build_all_goldens([
    HappyPathGoldens(),
    EdgeCaseGoldens(),
    AdversarialGoldens(),
])

dataset = EvaluationDataset(goldens=all_goldens)

def build_all_goldens(factories: list[BaseGoldenFactory]) -> list[Golden]:
    # this function only knows about the ABSTRACT BaseGoldenFactory interface —
    # it has no idea, and doesn't need to know, which concrete subclasses it's calling
    goldens = []
    for factory in factories:
        goldens.extend(factory.build_goldens())
    return goldens

all_goldens = build_all_goldens([
    HappyPathGoldens(),
    EdgeCaseGoldens(),
    AdversarialGoldens(),
    MultiTopicGoldens(),   # <-- the only change needed anywhere in existing code
])

dataset = EvaluationDataset(goldens=all_goldens)
# ... rest of your existing dataset.add_test_case() loop and parametrized test stay the same ...

@deepeval.log_hyperparameters(model="gpt-4o-mini", prompt_template="v1-qa-support-bot")
def hyperparameters():
    # return any additional hyperparameters you want tracked alongside model/prompt_template
    return {
        "temperature": 0,
        "retriever": "naive-keyword-v1",
    }
