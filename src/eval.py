"""Evaluate the model with simple_evals/mmlu_eval.py."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from exp import get_result_dir
from mmmlu_metadata import nlp_final_languages, subtasks, subtasks_for_taskc
from util import (
    alp_variations,
    answer_variations,
    colon_variations,
    custom_extract_answer,
)


def get_acc(path: str, model="google"):
    """
    Get the score from the jsonl file.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        lines = f.readlines()
    responses = []
    for line in lines:
        response = json.loads(line)
        responses.append(response)

    scores = []
    for i, response in enumerate(responses):
        extracted_answer = custom_extract_answer(response, model=model)
        score = 1.0 if extracted_answer == chr(i + ord("A")) else 0.0
        scores.append(score)
    return {
        "mean": sum(scores) / len(scores),
        "unshuffled": scores[responses[0]["answer"]],
        "first": scores[0],
        "second": scores[1],
        "third": scores[2],
        "last": scores[-1],
        "max": max(scores),
        "min": min(scores),
    }


def get_acc_from_result_dir(path: str, model="google"):
    """
    Get the scores from all jsonl files in the directory.
    """
    scores = {
        "mean": [],
        "unshuffled": [],
        "first": [],
        "second": [],
        "third": [],
        "last": [],
        "max": [],
        "min": [],
    }
    for file in Path(path).rglob("*.jsonl"):
        score = get_acc(file, model=model)
        scores["mean"].append(score["mean"])
        scores["unshuffled"].append(score["unshuffled"])
        scores["first"].append(score["first"])
        scores["second"].append(score["second"])
        scores["third"].append(score["third"])
        scores["last"].append(score["last"])
        scores["max"].append(score["max"])
        scores["min"].append(score["min"])

    scores["mean"] = sum(scores["mean"]) / len(scores["mean"])
    scores["unshuffled"] = sum(scores["unshuffled"]) / len(scores["unshuffled"])
    scores["first"] = sum(scores["first"]) / len(scores["first"])
    scores["second"] = sum(scores["second"]) / len(scores["second"])
    scores["third"] = sum(scores["third"]) / len(scores["third"])
    scores["last"] = sum(scores["last"]) / len(scores["last"])
    scores["max"] = sum(scores["max"]) / len(scores["max"])
    scores["min"] = sum(scores["min"]) / len(scores["min"])
    scores["min-max"] = scores["max"] - scores["min"]
    # std among first, second, third, last
    scores["rstd"] = float(
        np.std([scores["first"], scores["second"], scores["third"], scores["last"]])
    )
    # plot first, second, third, last
    import matplotlib.pyplot as plt

    x = ["A", "B", "C", "D"]
    y = [scores["first"], scores["second"], scores["third"], scores["last"]]
    plt.bar(x, y)
    plt.xlabel("Gold Answer")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of when all answers are placed to certain positions")
    plt.savefig(f"{path}/acc.png")
    plt.clf()

    return scores


def fluctuation_rate(
    path: str,
    model="google",
) -> float:
    """
    Calculate the fluctuation rate between two lists of answers.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        lines = f.readlines()
    responses = []
    for line in lines:
        response = json.loads(line)
        responses.append(response)

    answers = []
    for response in responses:
        extracted_answer = custom_extract_answer(
            response,
            model=model,
        )
        answers.append(extracted_answer)
    if len(answers[0]) and len(answers[-1]):
        return 1 if ord(answers[0]) + ord(answers[-1]) - 2 * ord("A") != 3 else 0
    return 1


def get_fr_from_result_dir(
    dir: str,
    model="google",
):
    """
    Get the fluctuation rate from all jsonl files in the directory.
    """
    scores = []
    for file in Path(dir).rglob("*.jsonl"):
        score = fluctuation_rate(file, model=model)
        scores.append(score)
    if len(scores) == 0:
        return 0
    return sum(scores) / len(scores)


def get_prob(
    path: str,
    model="google",
) -> float:
    """
    Get the probability from the jsonl file.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        lines = f.readlines()
    responses = []
    for line in lines:
        response = json.loads(line)
        responses.append(response)

    probs = []
    for _, response in enumerate(responses):
        extract_answer = custom_extract_answer(response, model=model)
        if model == "google":
            avg_lopgprob = response["candidates"][0]["avg_logprobs"]
            probs.append(math.exp(avg_lopgprob))
        elif model == "local":
            token_text_and_probs = response["token_text_and_probs"]
            for j, token in enumerate(token_text_and_probs):
                if (
                    j > 1
                    and token_text_and_probs[j - 1][0] in colon_variations
                    and token_text_and_probs[j - 2][0]
                    in answer_variations + [" Answer"]
                    and token[0][1] in set(alp_variations)
                ) or (
                    j > 2
                    and token_text_and_probs[j - 2][0] in colon_variations
                    and token_text_and_probs[j - 3][0]
                    in answer_variations + [" Answer"]
                    and (
                        token[0][0] in set(alp_variations)
                        or (len(token[0]) > 1 and token[0][1] in set(alp_variations))
                    )
                ):
                    probs.append(token[1])
                    break
            else:
                probs.append(0.0)
                if extract_answer != "":
                    print(token_text_and_probs[-10:])

    return sum(probs) / len(probs)


def get_prob_from_result_dir(dir: str, model="google"):
    """
    Get the probability from all jsonl files in the directory.
    """
    probs = []
    cnt = 0
    for file in Path(dir).rglob("*.jsonl"):
        prob = get_prob(file, model=model)
        if prob == 0.0:
            cnt += 1
        probs.append(prob)

    if len(probs) == 0:
        return 0
    return sum(probs) / len(probs)


def gather_scores(
    langs,
    subjects,
    model,
    prompt_method,
):
    """
    Generate a common dataframe for all languages and subjects.
    """
    results = {}
    for metric in ["mean", "rstd"]:
        scores = {lang: {} for lang in langs}

        for lang in langs:
            for subject in subjects:
                path = get_result_dir(
                    lang=lang,
                    subject=subject,
                    model=model,
                    prompt_method=prompt_method,
                )
                if not Path(path).exists():
                    continue
                acc = get_acc_from_result_dir(path, model=model)
                scores[lang][subject] = acc[metric]
        results[metric] = scores

    results["fr"] = {}
    results["prob"] = {}
    for lang in langs:
        results["fr"][lang] = {}
        results["prob"][lang] = {}
        for subject in subjects:
            path = get_result_dir(
                lang=lang,
                subject=subject,
                model=model,
                prompt_method=prompt_method,
            )
            if not Path(path).exists():
                continue
            fr = get_fr_from_result_dir(path, model=model)
            results["fr"][lang][subject] = fr
            prob = get_prob_from_result_dir(path, model=model)
            results["prob"][lang][subject] = prob

    return results


def gen_common():
    root_dir = Path("csvs/common")
    root_dir.mkdir(parents=True, exist_ok=True)
    for model in [
        "local",
        "google",
    ]:
        scores = gather_scores(
            langs=[obj[0] for obj in nlp_final_languages.values()],
            subjects=subtasks,
            model=model,
            prompt_method="cot",
        )

        for metric in ["mean", "rstd", "fr", "prob"]:
            df = pd.DataFrame(scores[metric])
            df.to_csv(
                root_dir / f"{model}_cot_{metric}.csv",
                index=True,
                header=True,
                sep=",",
            )


def gen_task_c():
    root_dir = Path("csvs/taskc")
    root_dir.mkdir(parents=True, exist_ok=True)

    for model in ["local", "google"]:
        cot_scores = gather_scores(
            langs=["EN_US", "JA_JP"],
            subjects=subtasks_for_taskc.values(),
            model=model,
            prompt_method="cot",
        )

        multichoice_scores = gather_scores(
            langs=["EN_US", "JA_JP"],
            subjects=subtasks_for_taskc.values(),
            model=model,
            prompt_method="multichoice",
        )

        for metric in ["mean", "rstd", "fr", "prob"]:
            cot_df = pd.DataFrame(cot_scores[metric])
            multichoice_df = pd.DataFrame(multichoice_scores[metric])
            cot_df.to_csv(
                root_dir / f"{model}_cot_{metric}.csv",
                index=True,
                header=True,
                sep=",",
            )
            multichoice_df.to_csv(
                root_dir / f"{model}_multichoice_{metric}.csv",
                index=True,
                header=True,
                sep=",",
            )

            diff_df = multichoice_df - cot_df
            diff_df.to_csv(
                root_dir / f"{model}_diff_{metric}.csv",
                index=True,
                header=True,
                sep=",",
            )


gen_common()
gen_task_c()
