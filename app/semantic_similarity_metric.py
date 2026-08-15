import os

import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

from deepeval.metrics import BaseMetric

from deepeval.test_case import LLMTestCase
from llm_config import get_openrouter_api_key, get_openrouter_base_url

load_dotenv()

client = OpenAI(
    api_key=get_openrouter_api_key(),
    base_url=get_openrouter_base_url(),
)

def get_embedding(text: str) -> np.ndarray:

    response = client.embeddings.create(
        model=os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        input=text,
    )

    return np.array(response.data[0].embedding)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:

    # exactly the formula from your Day 3 hand-calculation: dot product / (magnitude_a * magnitude_b)

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class SemanticSimilarityMetric(BaseMetric):

    def __init__(self, threshold: float = 0.75):

        self.threshold = threshold

        self.score = 0.0

        self.success = False

        self.reason = None

        self.error = None

    def measure(self, test_case: LLMTestCase) -> float:

        try:

            actual_embedding = get_embedding(test_case.actual_output)

            expected_embedding = get_embedding(test_case.expected_output)

            self.score = cosine_similarity(actual_embedding, expected_embedding)

            self.success = self.score >= self.threshold

            self.reason = f"Cosine similarity between actual and expected output: {self.score:.3f} (threshold: {self.threshold})"

            return self.score

        except Exception as e:

            self.error = str(e)

            self.success = False

            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:

        # async version required by DeepEval's concurrent evaluation (Day 7!) — reuses the sync logic for now

        return self.measure(test_case)

    def is_successful(self) -> bool:

        if self.error is not None:

            self.success = False

        else:

            try:

                self.success = self.score >= self.threshold

            except TypeError:

                self.success = False

        return self.success

    @property

    def __name__(self):

        return "Semantic Similarity"
