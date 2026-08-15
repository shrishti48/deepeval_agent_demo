from abc import ABC, abstractmethod
from deepeval.dataset import Golden

class BaseGoldenFactory(ABC):
    """Base class — every golden factory must implement build_goldens().
    This is your 'BasePage' equivalent."""

    @abstractmethod
    def build_goldens(self) -> list[Golden]:
        raise NotImplementedError


class HappyPathGoldens(BaseGoldenFactory):
    """Direct, single-fact lookups — the easy, expected-to-pass cases."""

    def build_goldens(self) -> list[Golden]:
        return [
            Golden(input="What if these shoes don't fit?", expected_output="You're eligible for a 30-day full refund at no extra cost."),
            Golden(input="How long does standard shipping take?", expected_output="Standard shipping takes 3-5 business days."),
            Golden(input="How do I reset my password?", expected_output="Click 'Forgot Password' on the login page and follow the emailed instructions."),
            Golden(input="Do gift cards expire?", expected_output="No, gift cards do not expire, but they can't be redeemed for cash."),
        ]


class EdgeCaseGoldens(BaseGoldenFactory):
    """Multi-fact, ambiguous, or boundary-condition questions."""

    def build_goldens(self) -> list[Golden]:
        return [
            Golden(input="If I return a defective item, when do I get my money back?", expected_output="Refunds are processed within 5-7 business days after the item is received."),
            Golden(input="Can I cancel my order?", expected_output="Only within 1 hour of placing the order, before it enters processing."),
            Golden(input="Why is my account locked?", expected_output="Accounts lock automatically after 5 failed login attempts, and unlock after 30 minutes."),
        ]


class AdversarialGoldens(BaseGoldenFactory):
    """Questions designed to trigger hallucination or expose retrieval weaknesses —
    the 'not in the knowledge base' and paraphrase-robustness cases from Day 10."""

    def build_goldens(self) -> list[Golden]:
        return [
            Golden(input="Do you ship to Antarctica?", expected_output="The bot should say it doesn't have this information, NOT invent a policy."),
            Golden(input="What's your policy on price matching with competitors?", expected_output="The bot should say it doesn't have this information."),
            Golden(input="My package still hasn't shown up after a week, is that normal?", expected_output="Should flag this as outside the normal 3-5 day shipping window, not just restate the policy blindly."),
        ]

class MultiTopicGoldens(BaseGoldenFactory):
    """Questions mixing two unrelated topics — tests whether the bot
    handles compound questions or only answers part of them."""

    def build_goldens(self) -> list[Golden]:
        return [
            Golden(
                input="How do I reset my password, and how long does shipping usually take?",
                expected_output="Should answer BOTH parts: password reset via 'Forgot Password' link, and 3-5 business days for standard shipping.",
            ),
        ]