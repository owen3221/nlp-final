import re
from pathlib import Path

MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    r"(?i){}[ \t]*([A-H]|[أ-ح]|[অএফগহ]|[ব]|[ড]|[ঢ]|[Ａ-Ｈ])"
)

# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Answer\s*:​​​​​​",  # Korean invisible character
    r"উত্তর\s*:",
    r"उत्तर\s*:",
    r"উত্তরঃ",
    r"উত্তর\s*:",
    r"Antwort\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"답\s*:",
    r"答案\s*：",
    r"答案\s*:",
    r"答\s*：",
    r"答\s*:",
    r"答复\s*：",
    r"答曰\s*：",
    r"الإجابة:",
    r"الجواب:",
    r"إجابة:",
    r"الإجابة النهائية:",
    r"الإجابة الصحيحة:",
    r"الإجابة الصحيحة هي:",
    r"الإجابة هي:",
    r"الجواب النهائي:",
    r"Respuesta\s*:",
    r"Risposta\s*:",
    r"答え\s*:",
    r"答え\s*：",
    r"回答\s*:",
    r"回答\s*：",
    r"解答\s*:",
    r"Jawaban\s*:",
    r"Réponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
    r"Ìdáhùn\s*:",
    r"Idáhùn\s*:",
    r"Àmọ̀nà\s*:",
    r"Àdáhùn\s*:",
    r"Ànúgọ\s*:",
    r"Àṣàyàn\s*:",
]


def normalize_response(response: str) -> str:
    """
    Normalize the response by removing markdown and LaTeX formatting that may prevent a match.
    """

    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


answer_variations = [
    "Answer",
    "Answer",
    "উত্তর",
    "उत्तर",
    "উত্তর",
    "উত্তর",
    "Antwort",
    "답변",
    "정답",
    "답",
    "答案",
    "答案",
    "答",
    "答",
    "答复",
    "答曰",
    "الإجابة",
    "الجواب",
    "إجابة",
    "الإجابة",
    "الإجابة",
    "الإجابة",
    "الإجابة",
    "الجواب",
    "Respuesta",
    "Risposta",
    "答え",
    "答え",
    "回答",
    "回答",
    "解答",
    "Jawaban",
    "Réponse",
    "Resposta",
    "Jibu",
    "Idahun",
    "Ìdáhùn",
    "Idáhùn",
    "Àmọ̀nà",
    "Àdáhùn",
    "Ànúgọ",
    "Àṣàyàn",
]
colon_variations = [
    ":",
    ":​​​​​​",  # with invisible character (e.g., U+2060 or similar)
    "：",  # full-width colon used in CJK
]
alp_variations = [
    # Arabic
    "أ",
    "ب",
    "ج",
    "د",
    "هـ",
    "و",
    "ز",
    "ح",
    # Bengali
    "অ",
    "ব",
    "ড",
    "ঢ",
    "এ",
    "ফ",
    "গ",
    "হ",
    # Japanese full-width
    "Ａ",
    "Ｂ",
    "Ｃ",
    "Ｄ",
    "Ｅ",
    "Ｆ",
    "Ｇ",
    "Ｈ",
    # Normal Alphabet
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
]


def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # Arabic letters for A–H (based on order)
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        .replace("هـ", " E")
        .replace("و", " F")
        .replace("ز", " G")
        .replace("ح", " H")
        # Bengali letters for A–H (based on phonetic approximations)
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        .replace("এ", " E")
        .replace("ফ", " F")
        .replace("গ", " G")
        .replace("হ", " H")
        # Japanese full-width characters for A–H
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .replace("Ｅ", " E")
        .replace("Ｆ", " F")
        .replace("Ｇ", " G")
        .replace("Ｈ", " H")
        .strip()
    )


def custom_extract_answer(response: dict, model="google") -> str:
    # 1) PriDe 结果优先
    if "pred_idx" in response:
        idx = int(response["pred_idx"])
        return chr(ord("A") + idx)

    # 2) 其他方法：从模型返回的文本里提取
    if model == "google":
        try:
            raw = response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            raw = response.get("generated_text", "")
    else:
        raw = response.get("generated_text", "")

    text = normalize_response(raw)

    # 3) 用多语言正则找答案字母
    extracted = ""
    for regex in MULTILINGUAL_ANSWER_REGEXES:
        pattern = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(regex)
        m = re.search(pattern, text)
        if m:
            extracted = normalize_extracted_answer(m.group(1))
            break

    if not extracted:
        return ""

    # 4) 如果是 E–H（multilingual 八选），映射回 A–D
    if extracted in ["E","F","G","H"]:
        extracted = chr(ord(extracted) - 4)

    return extracted

def get_result_dir(lang: str, subject: str, model: str, prompt_method: str) -> Path:
    """
    Get the result directory for the given language and subject.
    """
    # Return the path to the result directory
    return Path(f"results/{model}/{prompt_method}/{lang}/{subject}")
