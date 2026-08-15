# Describe the purpose of this module for readers opening the file directly.
"""
# Name the module in the header block for quick orientation.
rag_agent.py
# Draw a simple underline to separate the title from the description.
============
# Summarize the application as a compact RAG QA bot used for DeepEval practice.
A small RAG QA bot you can evaluate with DeepEval.
#
# Introduce the intentionally simple design choices used in this teaching example.
This keeps the setup intentionally simple for practice:
# Explain that the knowledge base is just a small in-memory document list.
  - Knowledge base: 9 policy snippets stored in memory
# Explain that retrieval uses lightweight local embeddings and an in-memory vector store.
  - Retrieval: local hash embeddings + InMemoryVectorStore
# Explain that the single tool performs search and returns the top matching chunks.
  - Tool: search_policies(query) returns the top 3 matching chunks
# Explain that the language model comes from the same OpenRouter config as the rest of the repo.
  - LLM: the same OpenRouter-backed chat model used elsewhere in the repo
# Explain that DeepEval tracing is wired through a callback and trace updates.
  - Tracing: DeepEval CallbackHandler + update_current_trace(...)
#
# Clarify that the target here is learnability rather than production-grade retrieval quality.
The goal is not production-grade retrieval. The goal is to give you a concrete
# Continue that rationale by naming the exact RAG metrics this bot supports practicing.
RAG system with retrieval_context so you can practice Answer Relevancy,
# Finish the module description with the remaining metrics.
Faithfulness, and Contextual Precision/Recall.
"""
#
# Import dotenv loading so local environment variables are available before model setup.
from dotenv import load_dotenv
#
# Load environment variables from .env as soon as the module imports.
load_dotenv()
#
# Import LangChain's tool decorator to expose the search function to the agent.
from langchain_core.tools import tool
# Import the in-memory vector store used for lightweight retrieval.
from langchain_core.vectorstores import InMemoryVectorStore
#
# Import the DeepEval LangChain callback that captures agent spans automatically.
from deepeval.integrations.langchain import CallbackHandler
# Import the helper used to write output and retrieval context onto the active trace.
from deepeval.tracing.context import update_current_trace
# Import the local embedding implementation and shared chat-model builder.
from llm_config import LocalHashEmbeddings, build_chat_model
# Import the helper that constructs a ReAct-style agent around the tool list.
from langgraph.prebuilt import create_react_agent
#
#
# Mark the beginning of the static knowledge-base section.
# ---------------------------------------------------------------------------
# Explain that the following strings are the policy documents the bot can retrieve from.
# Knowledge base - product and shipping policies as plain-text documents.
# Mark the end of the section heading.
# ---------------------------------------------------------------------------
# Store the support policy snippets in a list that will be indexed into the vector store.
POLICY_DOCS = [
    # Label the first subset as refund policies.
    # Refund policies
    # Add the electronics refund policy text as one logical document.
    "Electronics refund policy: Electronics can be returned within 15 days of delivery, "
    # Continue the same electronics policy document on the next source line.
    "provided the item is unopened and in its original packaging. Opened electronics are "
    # Finish the electronics policy with the non-returnable and refund timing details.
    "non-returnable unless faulty. Refunds are processed within 5-7 business days.",
    #
    # Add the clothing refund policy text as one logical document.
    "Clothing refund policy: Clothing items can be returned within 30 days of delivery "
    # Continue the clothing policy with the tag and condition requirements.
    "with all original tags still attached. Items must be unworn and unwashed. "
    # Finish the clothing policy with refund timing details.
    "Refunds are issued to the original payment method within 3-5 business days.",
    #
    # Add the food refund policy text as one logical document.
    "Food refund policy: Food and perishable items are non-returnable for health and "
    # Continue the food policy with the damaged or spoiled order handling rule.
    "safety reasons. If your food order arrived damaged or spoiled, contact support "
    # Finish the food policy with the refund or replacement outcome.
    "within 24 hours and we will issue a full refund or replacement.",
    #
    # Add the furniture refund policy text as one logical document.
    "Furniture refund policy: Furniture can be returned within 30 days if unassembled "
    # Continue the furniture policy with the assembled-item restriction.
    "and in original packaging. Assembled furniture cannot be returned unless defective. "
    # Finish the furniture policy with the return-shipping responsibility.
    "Return shipping costs are the customer's responsibility.",
    #
    # Add the jewellery refund policy text as one logical document.
    "Jewellery refund policy: Jewellery and personalised items are non-returnable unless "
    # Finish the jewellery policy with the damage or incorrect-item exception.
    "received in a damaged or incorrect condition. Please inspect items upon delivery.",
    #
    # Label the next subset as shipping policies.
    # Shipping policies
    # Add the standard shipping policy text as one logical document.
    "Standard shipping policy: Standard shipping takes 5-7 business days. "
    # Continue the standard-shipping policy with the free-shipping threshold.
    "Orders over $50 qualify for free standard shipping. Tracking information is "
    # Finish the standard-shipping policy with the tracking-email detail.
    "emailed once the order ships.",
    #
    # Add the express shipping policy text as one logical document.
    "Express shipping policy: Express shipping takes 1-2 business days and costs $15. "
    # Finish the express-shipping policy with the same-day dispatch cutoff.
    "Express orders placed before 2 PM are dispatched the same day.",
    #
    # Add the international shipping policy text as one logical document.
    "International shipping policy: International orders ship via courier and take "
    # Continue the international policy with customs and tax responsibility.
    "10-21 business days. Customs duties and import taxes are the buyer's responsibility. "
    # Finish the international policy with the country-coverage detail.
    "We ship to over 50 countries.",
    #
    # Label the final subset as order-management policy content.
    # Order management
    # Add the order cancellation policy text as one logical document.
    "Order cancellation policy: Orders can be cancelled within 1 hour of placement "
    # Continue the cancellation policy with the fulfilment cutoff detail.
    "for a full refund. After 1 hour, the order enters fulfilment and cannot be "
    # Finish the cancellation policy with the support-contact note.
    "cancelled. Contact support immediately if you need to cancel.",
]
#
#
# Mark the beginning of the vector-store initialization section.
# ---------------------------------------------------------------------------
# Explain that the vector store is built once at import time for simplicity.
# Build the vector store once at import time.
# Mark the end of the section heading.
# ---------------------------------------------------------------------------
# Create the deterministic local embedding model used to index and search the documents.
embeddings = LocalHashEmbeddings()
# Create the in-memory vector store that will hold the embedded policy documents.
vector_store = InMemoryVectorStore(embedding=embeddings)
# Add all policy documents to the vector store so they are searchable by similarity.
vector_store.add_texts(POLICY_DOCS)
#
#
# Mark the beginning of the retrieval-tool section.
# ---------------------------------------------------------------------------
# Explain that the next function retrieves the top matching chunks for a query.
# Retrieval tool - returns top-3 policy chunks for a query.
# Explain that retrieval results are also cached for DeepEval trace reporting.
# Also stashes the retrieved text in a module-level list so the @observe
# Finish that note by naming the trace field that receives the retrieved chunks.
# wrapper can forward it to the DeepEval trace as retrieval_context.
# Mark the end of the section heading.
# ---------------------------------------------------------------------------
# Store the latest retrieved chunks so the outer request can attach them to the trace.
_last_retrieved: list[str] = []
#
#
# Expose the search function to the agent as a callable tool.
@tool
# Define the search tool that queries the vector store for relevant policy documents.
def search_policies(query: str) -> str:
    # Describe the tool purpose for the model and for readers.
    """Search the customer-support knowledge base for policy information."""
    # Declare that the function updates the module-level retrieval cache.
    global _last_retrieved
    # Perform similarity search and return the top three matching documents.
    docs = vector_store.similarity_search(query, k=3)
    # Extract the raw text from the retrieved document objects.
    chunks = [doc.page_content for doc in docs]
    # Save the retrieved chunks so they can later be attached to the DeepEval trace.
    _last_retrieved = chunks
    # Return the retrieved chunks as a single tool response separated by blank lines.
    return "\n\n".join(chunks)
#
#
# Mark the beginning of the agent-construction section.
# ---------------------------------------------------------------------------
# Label this section as the agent setup block.
# Agent
# Mark the end of the section heading.
# ---------------------------------------------------------------------------
# Build the shared chat model with temperature zero for deterministic behavior.
llm = build_chat_model(temperature=0)
#
# Create the ReAct agent that can call the search tool before answering.
agent = create_react_agent(
    # Pass the chat model that will reason and generate answers.
    llm,
    # Register the policy-search tool as the only tool available to the agent.
    [search_policies],
    # Supply the system prompt that constrains the bot to answer from retrieved context.
    prompt=(
        # Tell the model what role it should play.
        "You are a policy QA bot for a small ecommerce knowledge base. "
        # Force the model to use retrieval before attempting an answer.
        "Always use the search_policies tool before answering. "
        # Restrict the model to the retrieved evidence.
        "Answer only from the retrieved information. "
        # Tell the model how to behave when the knowledge base lacks the answer.
        "If the answer is not covered, say that the knowledge base does not mention it. "
        # Keep the final answer style concise.
        "Keep replies concise."
    ),
)
#
# Explain that the callback captures the internal LangChain spans for DeepEval.
# DeepEval callback handler - captures LangChain spans automatically.
# Instantiate the shared callback once so each invocation can reuse it.
deepeval_callback = CallbackHandler()
#
#
# Mark the beginning of the public entry-point section.
# ---------------------------------------------------------------------------
# Label this section as the callable app entry point.
# Entry point
# Mark the end of the section heading.
# ---------------------------------------------------------------------------
# Define the main function that answers one user question with the RAG bot.
def rag_qa_bot(user_input: str) -> str:
    # Describe the function at a high level for readers and tooling.
    """Answer a user question from the policy knowledge base."""
    # Declare that the function resets and updates the module-level retrieval cache.
    global _last_retrieved
    # Clear any previous retrieval results before this new request starts.
    _last_retrieved = []
    #
    # Invoke the agent with a single user message and DeepEval callback tracing enabled.
    result = agent.invoke(
        # Pass the user input in the message format expected by the agent.
        {"messages": [{"role": "user", "content": user_input}]},
        # Attach the DeepEval callback so tool calls and LLM spans are recorded.
        config={"callbacks": [deepeval_callback]},
    )
    # Read the final assistant reply from the last message in the returned transcript.
    reply = result["messages"][-1].content
    #
    # Explain that the next trace update writes the final output and retrieval context.
    # Set clean reply string and retrieved chunks on the trace.
    # Push the answer and the retrieved chunks into the current DeepEval trace.
    update_current_trace(
        # Record the final answer text as the trace output.
        output=reply,
        # Record retrieval context when available, otherwise leave it unset.
        retrieval_context=_last_retrieved if _last_retrieved else None,
    )
    #
    # Return the final reply to the caller.
    return reply
#
#
# Explain that this alias preserves compatibility with older imports in the repo.
# Backwards-compatible alias for older eval files or notes in this repo.
# Point the legacy name at the new primary function.
rag_support_agent = rag_qa_bot
#
#
# Run a tiny smoke test when this module is executed directly.
if __name__ == "__main__":
    # Print a sample answer so the bot can be sanity-checked from the command line.
    print(rag_qa_bot("What is the return policy for electronics?"))
