"""Prompt templates for mmlu tasks."""

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


METHOD2_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDEFGH. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
E) {A}
F) {B}
G) {C}
H) {D}
""".strip()

PRIDE_TEMPLATE = """
Question: {Question}

A) {A}
B) {B}
C) {C}
D) {D}

Please output four probabilities that sum to 1.0. Do not assign 1.0 to any single option unless you are truly 100% certain.
Please output **only** the probability of each option, in the exact format below, with no explanation or extra text:

A: <probability between 0 and 1>
B: <probability between 0 and 1>
C: <probability between 0 and 1>
D: <probability between 0 and 1>
""".strip()

