"""Main experiment file."""

import json
import sys
import time
from pathlib import Path

import click
from tqdm import tqdm

sys.path.append("./src")
from data import gen_common, gen_taskc, get_experiment_dataset
from gen import failed_keys, get_response_google
from prompt import QUERY_TEMPLATE_MULTICHOICE


def get_result_dir(lang: str, subject: str, model: str) -> Path:
    """
    Get the result directory for the given language and subject.
    """
    # Return the path to the result directory
    return Path(f"results/{model}/{lang}/{subject}")


def save_responses(
    model: str,
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
    # Create the directory if it doesn't exist
    result_dir = get_result_dir(lang, subject, model)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save the requests to a jsonl file
    for request, response in zip(requests, responses):
        response["request"] = request
        response["answer"] = answer

    # Save the responses to a jsonl file
    with (result_dir / f"{index}.jsonl").open("w", encoding="utf-8") as f:
        # Write the responses to the file
        for response in responses:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")


def get_request_text(
    question: str,
    choices: list[str],
    answer: int,
) -> list[str]:
    """
    Generate the request text for the Gemini model.
    Returns:
        The index of the return list is the index of the answer in the choices.
    """
    # shuffle the choices based on the answer
    answer = choices[answer]
    non_answer_choices = [choice for choice in choices if choice != answer]
    if len(non_answer_choices) != 3:
        non_answer_choices.append("")
    shuffled_choices = [
        [answer] + non_answer_choices,
        [non_answer_choices[0], answer] + non_answer_choices[1:],
        non_answer_choices[:2] + [answer] + non_answer_choices[2:],
        non_answer_choices + [answer],
    ]
    return [
        QUERY_TEMPLATE_MULTICHOICE.format(
            Question=question,
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )
        for choices in shuffled_choices
    ]


def create_exp(dataset, lang: str, model="google") -> None:
    """
    Create the experiment for the given dataset and language.
    """
    for i, (question, subject, choices, answer) in enumerate(
        zip(
            dataset["question"],
            dataset["subject"],
            dataset["choices"],
            dataset["answer"],
        )
    ):
        # check if the directory exists
        result_dir = get_result_dir(
            lang=lang,
            subject=subject,
            model=model,
        )
        file_path = result_dir / f"{i}.jsonl"
        if file_path.exists():
            print(f"File {file_path} already exists. Skipping...")
            continue

        start = time.time()
        responses = []
        requests = get_request_text(question, choices, answer)

        for request in requests:
            response = get_response_google([request])
            print(response)
            print("\n\n!!!!You've successfully create the environment.!!!!\n\n")
            exit()
            responses.append(response)
            if model != "google":
                time.sleep(3)
        save_responses(
            model=model,
            requests=requests,
            responses=responses,
            lang=lang,
            subject=subject,
            answer=answer,
            index=i,
        )
        end = time.time()
        if model == "google":
            if end - start < 60 / 15 * 4:
                time.sleep(16 - (end - start) + 1)


@click.command()
@click.option(
    "--model",
    type=click.Choice(["google", "mistral", "nv"]),
    default="google",
    help="Model to use for the experiment.",
)
def main(
    model: str,
):
    """
    Main function to run the experiment.
    """
    # Generate common datasets
    datasets = gen_common()
    # datasets = gen_taskc()

    print("Using model:", model)

    for lang, subject in tqdm(datasets):
        # Start the experiment
        print(f"Language: {lang}, Subject: {subject}")
        print("Failed keys:", failed_keys)
        dataset = get_experiment_dataset(lang, subject)
        create_exp(dataset, lang, model=model)

    print("Experiment completed.")


if __name__ == "__main__":
    main()
