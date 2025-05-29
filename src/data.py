"""Data loading and processing for the MMMLU dataset."""

from datasets import load_dataset

from mmmlu_metadata import (
    nlp_final_languages,
    nlp_final_subcategories,
    subtasks,
    subtasks_for_taskc,
)


def transform_openai_mmmlu_to_cais_mmlu(dataset):
    questions = dataset["Question"]
    answers = dataset["Answer"]
    choices_a = dataset["A"]
    choices_b = dataset["B"]
    choices_c = dataset["C"]
    choices_d = dataset["D"]

    ret_questions = []
    ret_answers = []
    ret_choices = []

    for q, a, a_opt, b_opt, c_opt, d_opt in zip(
        questions, answers, choices_a, choices_b, choices_c, choices_d
    ):
        ret_questions.append(q)
        ret_choices.append([a_opt, b_opt, c_opt, d_opt])
        # Convert letter answer to index
        ret_answers.append(ord(a.upper()) - ord("A"))
    return {
        "question": ret_questions,
        "answer": ret_answers,
        "choices": ret_choices,
        "subject": dataset["Subject"],
    }


def get_experiment_dataset(
    lang: str = "JA_JP", subject: str = "abstract_algebra", num_samples: int = 20
):
    """
    Get a dataset for a specific language and subject from the MMMLU dataset.
    Args:
        lang (str): Language code (e.g., "JA_JP").
        subject (str): Subject name (e.g., "abstract_algebra").
        num_samples (int): Number of samples to return.
    Returns:
        Dataset or dict: A filtered dataset containing samples for the specified language and subject.
    """
    if lang == "EN_US":
        dataset = load_dataset("cais/mmlu", subject)["test"]
        return dataset[:num_samples]

    dataset = load_dataset("openai/MMMLU", lang)["test"]
    cat_ds = dataset.filter(lambda x: x["Subject"] == subject)
    return transform_openai_mmmlu_to_cais_mmlu(cat_ds[:num_samples])


def gen_common() -> list[tuple[str, str]]:
    """
    Generate a list of tuples containing language and subject pairs for the common 2 experiment.
    Only choose languages defined in nlp_final_languages for common testing.
    """
    langs = [obj[0] for obj in nlp_final_languages.values()]
    return [(lang, subtask) for subtask in subtasks for lang in langs]


def gen_taskc() -> list[tuple[str, str]]:
    """
    Generate a list of tuples containing language and subject pairs for the task C experiment.
    Only choose EN_US and JA_JP for task C testing.
    """
    subtasks_c = subtasks_for_taskc.values()
    return [(lang, subtask) for subtask in subtasks_c for lang in ["EN_US", "JA_JP"]]