from deepeval.metrics import GEval

from deepeval.test_case import SingleTurnParams

# --- Custom Faithfulness: stricter than the built-in, penalizes contradictions heavily ---

custom_faithfulness_metric = GEval(

    name="Strict Faithfulness",

    evaluation_steps=[

        "Identify every factual claim made in 'actual output'.",

        "For each claim, check whether it is directly supported by 'retrieval context'.",

        "If ANY claim contradicts or is unsupported by the retrieval context, this is a severe failure — score very low, even if only one claim is wrong.",

        "Do not give partial credit for an otherwise well-written answer if it contains even one unsupported factual claim.",

    ],

    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],

    threshold=0.7,

)

# --- Custom Relevance: focused specifically on whether the input's actual question was addressed ---

custom_relevance_metric = GEval(

    name="Custom Relevance",

    evaluation_steps=[

        "Identify the specific question or request being asked in 'input'.",

        "Check whether 'actual output' directly addresses that specific question, not just the general topic.",

        "Penalize answers that are topically related but don't actually answer what was asked (e.g., answering about shipping speed when asked about shipping cost).",

        "Do not penalize for brevity if the actual question is fully answered concisely.",

    ],

    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],

    threshold=0.7,

)

# --- Custom Coherence: no built-in equivalent — structure/flow, independent of factual correctness ---

custom_coherence_metric = GEval(

    name="Coherence",

    evaluation_steps=[

        "Check whether 'actual output' flows logically from one sentence to the next.",

        "Check whether the response contradicts itself anywhere within its own text.",

        "Check whether the response stays focused, without abruptly switching topics or trailing off incompletely.",

        "Evaluate structure and flow ONLY — do not factor in whether the content is factually correct; a coherent answer can still be wrong, and a correct answer can still be incoherent.",

    ],

    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],

    threshold=0.7,

)

custom_tonality_metric = GEval(
    name="Custom Tonality",
    evaluation_steps=[
        "Identify the specific question or request being asked in 'input'.",
        "Check whether 'actual output' directly addresses that specific question, not just the general topic.",
        "Penalize answers that are topically related but don't actually answer what was asked (e.g., answering about shipping speed when asked about shipping cost).",
        "Do not penalize for brevity if the actual question is fully answered concisely.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.7,
)