from dotenv import load_dotenv

from data.knowledge_base.knowledge_base import KNOWLEDGE_BASE
from llm_config import create_openrouter_chat_completion

load_dotenv()

def retrieve(query: str, top_k: int = 2) -> list[str]:

    # naive keyword-overlap "retriever" — good enough for a skeleton project

    scored = sorted(

        KNOWLEDGE_BASE,

        key=lambda doc: sum(1 for word in query.lower().split() if word in doc.lower()),

        reverse=True,

    )

    return scored[:top_k]

def generate(query: str, context: list[str]) -> str:

    context_str = "\n".join(context)

    response = create_openrouter_chat_completion(
        messages=[
            {"role": "system", "content": "Answer the user's question using ONLY the provided context. If the context doesn't contain the answer, say you don't have that information."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"},
        ],
        temperature=0,
    )

    return response.choices[0].message.content

def qa_bot(query: str) -> tuple[str, list[str]]:

    context = retrieve(query)

    answer = generate(query, context)

    return answer, context
