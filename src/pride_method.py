import re
import time
import math

import numpy as np
from tqdm import tqdm

from prompt import PRIDE_TEMPLATE
from gen import get_response_google
from util import get_result_dir  # 仅用于构造结果路径


def softmax(logits: np.ndarray) -> np.ndarray:
    """对数几率向量做 softmax，返回概率向量。"""
    ex = np.exp(logits - np.max(logits))
    return ex / ex.sum()


def extract_letter_logps(text: str, eps: float = 1e-8) -> dict[str, float]:
    """
    从模型返回的纯文本中解析每个字母的原始概率，再取 log:
      输入 text:
        A: 0.10
        B: 0.20
        C: 0.30
        D: 0.40
      返回 {'A': log(0.10), 'B': log(0.20), ...}
    p<=0 时 clamp 到 eps，避免 log(0).
    """
    logps = {}
    for letter in ("A", "B", "C", "D"):
        m = re.search(rf"{letter}:\s*([01](?:\.\d+)?)", text)
        p = float(m.group(1)) if m else 0.0
        if p <= 0.0:
            p = eps
        logps[letter] = math.log(p)
    return logps


def estimate_prior(
    D_e: list[tuple[str, list[str], int]],
    model: str = "google",
) -> np.ndarray:
    """
    Phase 1: 对前 K 道题做循环置换，多次调用模型，
    计算每个选项在原始位置上的平均 log-prob，再 softmax 得到先验分布。
    返回形如 [P_prior(A), P_prior(B), P_prior(C), P_prior(D)]。
    """
    all_priors = []
    for question, choices, _ in tqdm(D_e, desc="Estimating prior", unit="q"):
        # 对本题所有置换收集 remapped log-probs
        remapped_logits = []
        for shift in range(len(choices)):
            perm = choices[-shift:] + choices[:-shift]   # 右移 shift
            prompt = PRIDE_TEMPLATE.format(
                Question=question,
                A=perm[0], B=perm[1], C=perm[2], D=perm[3],
            )
            rsp = get_response_google([prompt])[0]
            time.sleep(60 / 15)

            text = rsp["candidates"][0]["content"]["parts"][0]["text"]
            letter_logps = extract_letter_logps(text)

            # remap 回原始 choices 顺序
            logps = []
            for orig in choices:
                k = perm.index(orig)                    # orig 在 perm 中的位置
                letter = chr(ord("A") + k)              # perm 中的那个字母
                logps.append(letter_logps[letter])
            remapped_logits.append(logps)

        avg_log = np.mean(remapped_logits, axis=0)
        all_priors.append(softmax(avg_log))

    # K 道题的先验再平均
    return np.mean(np.stack(all_priors, axis=0), axis=0)


def create_pride_exp(
    dataset: list[dict] | dict[str, list],
    lang: str,
    subject: str,
    model: str = "google",
    K: int = 5,
):
    """
    完整 PriDe 流程：
      1) Phase1: 用前 K 题 estimate_prior
      2) Phase2: 对剩余每题，对 4 个循环置换各自调用模型，
         remap→debiased logits→softmax→argmax→存 4 条 response
    """
    # 支持 dict-of-lists 输入
    if isinstance(dataset, dict):
        dataset = [
            {"question": q, "choices": c, "answer": a}
            for q, c, a in zip(
                dataset["question"],
                dataset["choices"],
                dataset["answer"],
            )
        ]

    # 拆成[(q,choices,ans),...]
    samples = [(d["question"], d["choices"], d["answer"]) for d in dataset]
    D_e, D_r = samples[:K], samples[K:]
    prior = estimate_prior(D_e, model)

    # Phase 2: 对每道题的每个循环置换都做一次推理，并 remap 回原始顺序
    for idx, (q, choices, ans) in enumerate(tqdm(D_r, desc="Debiasing", unit="q")):
        responses = []
        for shift in range(len(choices)):
            perm = choices[-shift:] + choices[:-shift]
            prompt = PRIDE_TEMPLATE.format(
                Question=q,
                A=perm[0], B=perm[1], C=perm[2], D=perm[3],
            )
            rsp = get_response_google([prompt])[0]
            time.sleep(60 / 15)

            # 1) 解析 letter→logp
            text = rsp["candidates"][0]["content"]["parts"][0]["text"]
            letter_logps = extract_letter_logps(text)

            # 2) remap 回原始顺序
            remapped = []
            for orig in choices:
                k = perm.index(orig)
                letter = chr(ord("A") + k)
                remapped.append(letter_logps[letter])

            # 3) 去偏 & softmax & argmax
            observed = np.array(remapped)
            debiased = observed - np.log(prior)
            final_probs = softmax(debiased)
            pred_idx = int(np.argmax(final_probs))

            responses.append({
                "request":      prompt,
                "raw_response": rsp,
                "pred_idx":     pred_idx,
                "final_probs":  final_probs.tolist(),
                "answer":       ans,
            })

        # 保存到 results/.../{idx}.jsonl
        from exp import save_responses
        save_responses(
            model=model,
            prompt_method="pride",
            requests=[r["request"] for r in responses],
            responses=responses,
            lang=lang,
            subject=subject,
            answer=ans,
            index=idx,
        )