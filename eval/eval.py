"""Evaluate the model with simple_evals/mmlu_eval.py."""

import json
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

# 确保能够 import 到 src 里的代码
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from exp import get_result_dir
from mmmlu_metadata import subtasks_for_taskc
from util import custom_extract_answer

def fluctuation_rate(ans1: list[str], ans2: list[str]) -> float:
    """
    Calculate the fluctuation rate (FR) between two lists of answers.
    FR 定义为答案不一致的比例。
    """
    if len(ans1) != len(ans2):
        raise ValueError("Answer lists must be the same length")
    diffs = sum(a1 != a2 for a1, a2 in zip(ans1, ans2))
    return diffs / len(ans1)

def get_fr_from_result_dir(path: str) -> float:
    """
    对一个子任务目录下的所有 jsonl 文件，计算平均的 Fluctuation Rate。
    对于每道题：
      - 先按文件顺序读取所有 response，抽出每条的预测字母
        （PriDe 用 pred_idx → 字母; CoT/Multichoice 用 custom_extract_answer）。
      - 以第一条（unshuffled）预测作为基线，计算其与后续三次的 FR。
    最后对所有题目的 FR 取平均。
    """
    frs = []
    for file in Path(path).rglob("*.jsonl"):
        responses = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
        # 抽出每条 response 的预测字母
        preds = []
        for resp in responses:
            if "pred_idx" in resp:
                preds.append(chr(resp["pred_idx"] + ord("A")))
            else:
                # CoT/Multichoice 分支
                preds.append(custom_extract_answer(resp))
        # baseline = preds[0] 重复三次，与 preds[1:] 比较
        baseline_list = [preds[0]] * len(preds)
        fr = fluctuation_rate(baseline_list, preds)
        frs.append(fr)
    return float(np.mean(frs)) if frs else 0.0

def get_acc(path: str, model="google"):
    """
    Get the score from the jsonl file.
    """
    responses = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]

    # —— PriDe 单 response 分支 —— #
    if len(responses) == 1 and "pred_idx" in responses[0]:
        resp   = responses[0]
        pred_i = int(resp.get("pred_idx", -1))
        true_i = int(resp.get("answer", -1))
        score  = 1.0 if pred_i == true_i else 0.0
        return {
            "mean": score,
            "unshuffled": score,
            "first": score,
            "second": score,
            "third": score,
            "last": score,
            "max": score,
            "min": score,
        }

    # —— CoT/Multichoice 原有分支 —— #
    scores = []
    for i, response in enumerate(responses):
        extracted = custom_extract_answer(response, model=model)
        correct   = chr(ord("A") + i)
        scores.append(1.0 if extracted == correct else 0.0)

    return {
        "mean":      sum(scores) / len(scores),
        "unshuffled": scores[responses[0]["answer"]],
        "first":     scores[0],
        "second":    scores[1],
        "third":     scores[2],
        "last":      scores[-1],
        "max":       max(scores),
        "min":       min(scores),
    }


def get_acc_from_result_dir(path: str, model="google"):
    """
    Get the scores from all jsonl files in the directory.
    """
    result_path = Path(path)
    metrics = {k: [] for k in ["mean","unshuffled","first","second","third","last","max","min"]}
    for file in result_path.rglob("*.jsonl"):
        sc = get_acc(str(file), model=model)
        for k in metrics:
            metrics[k].append(sc[k])

     # std among first, second, third, last
    agg = {k: sum(v)/len(v) for k, v in metrics.items()}
    agg["min-max"] = agg["max"] - agg["min"]
    agg["rstd"]    = float(np.std([agg["first"], agg["second"], agg["third"], agg["last"]]))

    # plot first, second, third, last
    import matplotlib.pyplot as plt
    x = ["A","B","C","D"]
    y = [agg["first"], agg["second"], agg["third"], agg["last"]]
    plt.bar(x, y)
    plt.xlabel("Gold Answer")
    plt.ylabel("Accuracy")
    plt.title("Positional Accuracy")
    # 确保目录存在
    plt_path = result_path / "acc.png"
    plt_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plt_path)
    plt.clf()

    return agg


def gen_common_df(langs, subjects, model, prompt_method):
    """
    生成 heatmap，并保存到根目录下的 pngs/。
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 确保 pngs/ 目录存在
    png_dir = ROOT / "pngs"
    png_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["mean", "rstd", "min-max"]:
        print(f"Metric: {metric}")
        data = {lang: {} for lang in langs}

        for lang in langs:
            for subject in subjects:
                result_dir = get_result_dir(
                    lang=lang,
                    subject=subject,
                    model=model,
                    prompt_method=prompt_method,
                )
                if not Path(result_dir).exists():
                    continue
                acc = get_acc_from_result_dir(result_dir, model=model)
                data[lang][subject] = acc[metric]

        df = pd.DataFrame(data)
        print(df)
        # 如果没有任何数据，就跳过这个 metric
        if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
            print(f"No data for {model}/{prompt_method}/{metric}, skipping heatmap.")
            continue
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues")
        plt.title(f"{model} / {prompt_method} / {metric}")
        plt.savefig(png_dir / f"{model}_{prompt_method}_{metric}_heatmap.png")
        plt.clf()


if __name__ == "__main__":
    gen_common_df(
        langs=["EN_US", "JA_JP"],
        subjects=subtasks_for_taskc.values(),
        model="google",
        prompt_method="pride",
    )

    gen_common_df(
    langs=["EN_US", "JA_JP"],
    subjects=subtasks_for_taskc.values(),
    model="google",
    prompt_method="cot",
    )