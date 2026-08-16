# DeepEval Learning Repo

This repo is not one polished production app. It is a study repo for learning how to evaluate LLM systems with DeepEval across four shapes of AI application:

- a plain tool-calling agent
- the same agent with tracing and evaluation hooks
- a simple single-turn RAG QA bot
- a multi-turn chatbot

The main value of the repo is that almost every file teaches one narrow lesson. Read it like a workbook, not like a library.

## What You Should Learn Here

By the time you finish this repo, you should be able to explain:

- what a `Golden` is and why it is different from a fully built `LLMTestCase`
- when to use `assert_test(...)`, `evaluate(...)`, and `evals_iterator(...)`
- how tracing changes DeepEval from “judge one final string” into “inspect the full app behavior”
- how RAG evals differ from plain agent evals
- why multi-turn evals need different data structures and metrics than single-turn evals
- how to write custom metrics when the built-in metrics are not enough
- how to keep model configuration in one place instead of scattering provider logic across tests

## Setup

Use the project virtualenv and install the requirements:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Create a local `.env` file with your OpenRouter settings:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-oss-20b:free
OPENROUTER_MODELS=openai/gpt-oss-20b:free,google/gemma-4-26b-a4b-it:free
OPENROUTER_JUDGE_MODELS=openai/gpt-oss-20b:free,google/gemma-4-26b-a4b-it:free
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

Important local notes:

- `.env.local` and other secret-bearing env files should not be committed.
- free OpenRouter models can hit daily rate limits quickly when you run many evals.
- some tests in this repo are intentionally rough teaching experiments, not all of them are hardened CI-grade suites.

## How To Read This Repo

Use this order. Each step builds the next.

1. Read [`agent_plain.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/agent_plain.py) to see the uninstrumented baseline.
2. Read [`agent_instrumented.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/agent_instrumented.py) to see the smallest useful DeepEval tracing change.
3. Read [`llm_config.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/llm_config.py) to understand how the whole repo routes models through OpenRouter.
4. Read [`app/qa_bot.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/app/qa_bot.py) for the simplest RAG-like flow: retrieve, then generate.
5. Read [`app/rag_agent.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/app/rag_agent.py) for a more agentic RAG version with retrieval context attached to traces.
6. Read [`chatbot.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/chatbot.py) for the multi-turn tool loop.
7. Then read the `evals/` directory file by file to see how each app shape gets tested.

## File-By-File Lessons

### Root app files

[`agent_plain.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/agent_plain.py)

- Lesson: this is what a developer would naturally build before evaluation exists.
- Learn: tools are just functions plus a prompt; there is no DeepEval dependency here.
- Pay attention to: the clean `support_agent(...)` entry point. Tests need one stable app boundary to call.

[`agent_instrumented.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/agent_instrumented.py)

- Lesson: tracing does not require rewriting the app.
- Learn: the repo’s core DeepEval pattern is `CallbackHandler()` plus `update_current_trace(...)`.
- Pay attention to: `tools_called` extraction. This is what makes `ToolCorrectnessMetric` possible later.

[`chatbot.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/chatbot.py)

- Lesson: multi-turn evaluation requires preserving history, not just final answers.
- Learn: single-turn agent frameworks can hide the tool loop; chatbot evals often need you to control it explicitly.
- Pay attention to: `history`, `tools_called`, and the repeated call loop until there are no more tool calls.

[`llm_config.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/llm_config.py)

- Lesson: centralize provider logic once.
- Learn: this file abstracts OpenRouter for three use cases:
  - chat completions for plain app code
  - LangChain `ChatModel` usage
  - DeepEval judge model usage
- Pay attention to:
  - fallback model rotation
  - rate-limit-aware retry/fallback logic
  - `LocalHashEmbeddings`, which keeps local retrieval cheap and deterministic

[`mcp_server.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/mcp_server.py)

- Lesson: evaluation is not limited to trivial local tools; the repo also hints at MCP-based tool expansion.
- Learn: once your app boundary is stable, the tool source can change without changing your evaluation mindset.

### RAG app files

[`app/qa_bot.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/app/qa_bot.py)

- Lesson: the simplest RAG bot is just `retrieve()` and `generate()`.
- Learn: you do not need a vector database on day one to practice DeepEval.
- Pay attention to:
  - naive keyword retrieval
  - explicit “answer only from context” prompting
  - return shape `(answer, context)` which is ideal for building `LLMTestCase`s

[`app/rag_agent.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/app/rag_agent.py)

- Lesson: a RAG system can also be agentic, not just one retrieval function.
- Learn: retrieval context must be surfaced into the trace if you want to use RAG metrics like Faithfulness, Contextual Precision, and Contextual Recall.
- Pay attention to:
  - the `search_policies` tool
  - `_last_retrieved` as the bridge between retrieval and trace data
  - `update_current_trace(output=..., retrieval_context=...)`

[`data/knowledge_base/knowledge_base.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/data/knowledge_base/knowledge_base.py)

- Lesson: a tiny static corpus is enough to practice RAG evaluation.
- Learn: your goldens should be derived from what is and is not in this knowledge base.

[`data/knowledge_base/goldens.json`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/data/knowledge_base/goldens.json)

- Lesson: goldens belong in data, not only inline in code.
- Learn: once goldens are file-backed, the same dataset can be reused across different metric suites.

### Custom metric files

[`app/custom_eval_metrics.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/app/custom_eval_metrics.py)

- Lesson: built-in metrics are a starting point, not the finish line.
- Learn: `GEval` becomes much stronger when you write explicit scoring instructions for your exact failure mode.
- Pay attention to:
  - strict faithfulness
  - question-specific relevance
  - coherence as a separate dimension from correctness

[`app/semantic_similarity_metric.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/app/semantic_similarity_metric.py)

- Lesson: not every metric has to be LLM-as-judge.
- Learn: a metric can be a normal Python class using embeddings plus cosine similarity.
- Pay attention to:
  - OpenRouter embeddings usage
  - `measure`, `a_measure`, and `is_successful`
  - the distinction between semantic closeness and exact wording

### Evals support files

[`evals/metrics.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/metrics.py)

- Lesson: keep metric bundles separate from the test file.
- Learn: this is the clean DeepEval pattern for reusable committed eval suites.

[`evals/golden_factories.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/golden_factories.py)

- Lesson: goldens can be organized by intent, not just stored in one giant list.
- Learn: separating happy-path, edge-case, adversarial, and multi-topic questions makes the dataset easier to reason about and extend.

### Single-turn eval files

[`evals/test_example.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_example.py)

- Lesson: the smallest possible RAG-like metric example.
- Learn: how `AnswerRelevancyMetric`, `FaithfulnessMetric`, and `HallucinationMetric` look on one manually built `LLMTestCase`.

[`evals/test_correctness.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_correctness.py)

- Lesson: correctness is often a custom judgment rubric, not a string match.
- Learn: `GEval` is flexible enough to encode nuanced domain instructions.

[`evals/test_reference_based.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_reference_based.py)

- Lesson: some evaluations are reference-based and embedding-based instead of LLM-judge-based.
- Learn: semantic similarity can complement judge metrics when wording variation is acceptable.

### Agent eval files

[`evals/test_TaskCompletion.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_TaskCompletion.py)

- Lesson: start with one metric on one case before scaling out.
- Learn: `evaluate(...)` is fine for tiny direct experiments.

[`evals/test_TracingComponentsTest.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_TracingComponentsTest.py)

- Lesson: tool-aware evaluation depends on tracing and expected tool metadata.
- Learn: DeepEval can judge not just the answer, but whether the app used the right tool.

[`evals/test_customMetricEvals.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_customMetricEvals.py)

- Lesson: expected output can be injected into the trace and judged later.
- Learn: trace enrichment is what makes higher-quality evals possible.

[`evals/test_multipleEvalsTest.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_multipleEvalsTest.py)

- Lesson: one app run can be evaluated against several dimensions at once.
- Learn: answer relevancy, prompt alignment, and step efficiency are different questions about the same behavior.

[`evals/test_agent_synthesized.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_agent_synthesized.py)

- Lesson: datasets do not always need to be hand-written.
- Learn: `Synthesizer.generate_goldens_from_docs(...)` is one path to bootstrapping eval data from docs.

### RAG eval files

[`evals/test_rag_agent.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_rag_agent.py)

- Lesson: this is the closest file in the repo to the “recommended DeepEval RAG pattern.”
- Learn:
  - load goldens from data
  - run the traced app
  - call `assert_test(golden=golden, metrics=RAG_TRACE_METRICS)`
- Pay attention to: how expected output is copied into the trace before the app call.

[`evals/test_rag_qa_bot.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_rag_qa_bot.py)

- Lesson: the no-tracing fallback is still useful when your app returns answer plus context directly.
- Learn:
  - build `LLMTestCase` inside the parametrized test
  - compare built-in metrics vs custom metrics
  - organize goldens by scenario
- Important: this file has gone through several teaching iterations; treat it as a live workshop file, not the cleanest final artifact in the repo.

### Multi-turn eval files

[`evals/test_chatbot.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_chatbot.py)

- Lesson: multi-turn metrics need conversation objects, not plain `LLMTestCase`.
- Learn: `ConversationalTestCase` plus `Turn` objects are the basic unit for chatbot evaluation.

[`evals/test_chatbot_customreqd.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/evals/test_chatbot_customreqd.py)

- Lesson: conversational correctness can also be framed with a custom rubric.
- Learn: `ConversationalGEval` lets you judge the whole dialogue rather than one answer.

## Recommended Run Order

Run these in increasing complexity.

```bash
.venv/bin/python agent_instrumented.py
.venv/bin/python app/qa_bot.py
.venv/bin/python chatbot.py
```

Then try evals in this order:

```bash
.venv/bin/deepeval test run evals/test_example.py
.venv/bin/deepeval test run evals/test_correctness.py
.venv/bin/deepeval test run evals/test_rag_agent.py
.venv/bin/deepeval test run evals/test_rag_qa_bot.py
```

## What This Repo Teaches About DeepEval

The repo repeats the same few ideas until they become obvious:

- keep one stable app entry point per system
- make traces rich enough for the metrics you want to use
- separate dataset, app execution, and metric definition
- write goldens that test failure modes, not just happy paths
- treat evaluation as code you rerun, inspect, and improve

## Current Caveats

- Some files are intentionally rough because they were built incrementally as learning exercises.
- Some evals make live OpenRouter calls and can fail on quota, rate limits, or network access.
- Some files still use `create_react_agent` from LangGraph and should eventually move to the newer `langchain.agents.create_agent` path.
- The heavily commented [`app/rag_agent.py`](/Users/shrishti/PycharmProjects/deepeval-agent-demo/app/rag_agent.py) is optimized for study, not for production readability.

## If You Want To Study “Line By Line”

Use this method:

1. Open one file.
2. For every import, ask: “why does this file need this dependency?”
3. For every function, ask: “is this app logic, trace logic, dataset logic, or metric logic?”
4. For every test, identify:
   - where input comes from
   - where actual output is generated
   - where expected output lives
   - which metric consumes which fields
5. After each file, summarize the single lesson it teaches in one sentence.

If you can do that for the files above, you understand the repo.
