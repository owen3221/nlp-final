"""Main experiment file."""

import json
import time

import click
from tqdm import tqdm

from data import gen_common, gen_taskc, get_experiment_dataset
from gen import get_response_google
from prompt import (
    METHOD2_TEMPLATE_MULTICHOICE,
    QUERY_TEMPLATE_MULTICHOICE,
    PRIDE_TEMPLATE,
)
from util import get_result_dir
from pride_method import create_pride_exp


def save_responses(
    model: str,
    prompt_method: str,
    requests: list[str],
    responses: list[dict],
    lang: str,
    subject: str,
    answer: int,
    index: int,
) -> None:
    """
    Save the responses to a jsonl file.
    """
    result_dir = get_result_dir(lang, subject, model, prompt_method)
    result_dir.mkdir(parents=True, exist_ok=True)

    for request, response in zip(requests, responses):
        response["request"] = request
        response["answer"] = answer

    with (result_dir / f"{index}.jsonl").open("w", encoding="utf-8") as f:
        for response in responses:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")


def format_request(
    question: str,
    choices: list[str],
    method: str = "cot",  # choices: "cot", "pride", "multilingual", "multichoice"
) -> str:
    """
    Format the request for the Gemini model.
    """
    if method == "cot":
        return QUERY_TEMPLATE_MULTICHOICE.format(
            Question=question,
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )
    elif method == "pride":
        print("choose pride")
        return PRIDE_TEMPLATE.format(
            Question=question,
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )
    elif method == "multilingual":
        # TODO: implement multilingual prompt method
        raise NotImplementedError("multilingual prompt method not implemented")
    elif method == "multichoice":
        return METHOD2_TEMPLATE_MULTICHOICE.format(
            Question=question,
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )
    else:
        raise ValueError(f"Unknown prompt_method: {method}")


def get_request_text(
    question: str,
    choices: list[str],
    answer: int,
    prompt_method: str = "cot",
) -> list[str]:
    """
    Generate the request text for the Gemini model, producing 4 permutations
    of choices so that correct answer appears in each position.
    """
    correct = choices[answer]
    others = [c for c in choices if c != correct]
    others += [""] * (3 - len(others))
    shuffled = [
        [correct, others[0], others[1], others[2]],
        [others[0], correct, others[1], others[2]],
        [others[0], others[1], correct, others[2]],
        [others[0], others[1], others[2], correct],
    ]
    return [
        format_request(question, perm, prompt_method)
        for perm in shuffled
    ]


def create_exp(
    dataset,
    lang: str,
    model="google",
    prompt_method="cot",
) -> None:
    """
    Create the experiment for the given dataset and language.
    """
    for i, (question, subject, choices, answer) in enumerate(
        tqdm(
            zip(
                dataset["question"],
                dataset["subject"],
                dataset["choices"],
                dataset["answer"],
            ),
            total=len(dataset["question"]),
        ),
    ):
        result_dir = get_result_dir(lang, subject, model, prompt_method)
        file_path = result_dir / f"{i}.jsonl"
        if file_path.exists():
            print(f"File {file_path} already exists. Skipping...")
            continue

        start = time.time()
        requests = get_request_text(
            question,
            choices,
            answer,
            prompt_method=prompt_method,
        )
        # 仅支持 Google/Gemini
        if model == "google":
            responses = get_response_google(requests)
        # elif model == "local":
        #     from local_gen import get_response_local
        #     responses = get_response_local(requests)
        else:
            raise ValueError(f"Unknown model: {model}")

        save_responses(
            model=model,
            prompt_method=prompt_method,
            requests=requests,
            responses=responses,
            lang=lang,
            subject=subject,
            answer=answer,
            index=i,
        )
        end = time.time()
        if model == "google":
            elapsed = end - start
            # rate limit: allow 15 calls/minute → ~4 calls per sample → 16s
            if elapsed < 16:
                time.sleep(16 - elapsed)


@click.command()
@click.option(
    "--model",
    type=click.Choice(["google"]),  # 只保留 google
    default="google",
    help="Model to use for the experiment.",
)
@click.option(
    "--prompt_method",
    type=click.Choice(["cot", "pride", "multilingual", "multichoice"]),
    default="cot",
    help="Prompt method to use for the experiment.",
)
@click.option(
    "--exp",
    type=click.Choice(["common", "taskc"]),
    default="common",
    help="Experiment to run.",
)
def main(
    model: str,
    prompt_method: str,
    exp: str,
):
    """
    Main function to run the experiment.
    """
    
    datasets = gen_common() if exp == "common" else gen_taskc()
    datasets = datasets[:1]#test

    print("Using model:", model)

    for lang, subject in tqdm(datasets):
        print(f"Language: {lang}, Subject: {subject}")
        dataset = get_experiment_dataset(lang, subject)
        if prompt_method == "pride":
            # PriDe 默认 K=5，已在 pride_method.py 里设置
            create_pride_exp(dataset, lang, subject, model)
        else:
            create_exp(
                dataset,
                lang,
                model=model,
                prompt_method=prompt_method,
            )

    print("Experiment completed.")


if __name__ == "__main__":
    main()