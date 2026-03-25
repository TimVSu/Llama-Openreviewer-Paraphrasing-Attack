import os
import math
import requests
from functools import reduce
import re
import nltk
import anthropic
import pandas as pd
import numpy as np
import time
from functools import wraps
import httpx
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache


def time_it(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function '{func.__name__}' took {end - start:.4f} seconds")
        return result

    return wrapper


def clean_markdown_latex(md_string: str) -> str:
    latex_patterns = [
        r"\$\$[\s\S]*?\$\$",
        r"\\\[[\s\S]*?\\\]",
        r"\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}",
        r"\$[^\$\n]+?\$",
        r"\\\([\s\S]*?\\\)",
    ]
    clean_md = reduce(
        lambda text, p: re.sub(p, "MATH_BLOCK", text), latex_patterns, md_string
    )

    skip_line = (
        lambda line: re.match(r"^#{1,6}\s", line.strip())
        or re.match(r"^[-*_]{3,}$", line.strip())
        or re.match(r"^\*{1,2}[^*]+\*{1,2}$", line.strip())
    )
    return "\n".join(line for line in clean_md.splitlines() if not skip_line(line))


def extract_sentences(cleaned_text: str) -> list[str]:
    nltk.download("punkt")
    nltk.download("punkt_tab")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned_text) if p.strip()]

    return [
        sentence
        for paragraph in paragraphs
        for sentence in nltk.sent_tokenize(paragraph)
    ]


def load_file_as_string(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        paper_text = f.read()
    return paper_text


def is_natural_prose(text):
    forbidden_symbols = [r"=⇒", r"6=⇒", r"=", r"~", r"\[", r"\]", r"\(wl"]
    if any(re.search(sym, text) for sym in forbidden_symbols):
        return False

    if re.search(r"\w+\s*,\s*\w+\s*(cl|cr|=)", text):
        return False

    words = re.findall(r"\w+", text)
    symbols = re.findall(r"[^\w\s]", text)

    if len(words) > 0 and (len(symbols) / len(words)) > 0.2:
        return False

    return True


def is_semantically_similar(text1, text2, semantic_treshold, sentence_transformer):
    embeddings = sentence_transformer.encode([text1, text2])
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return cosine_score >= semantic_treshold


def is_sufficiently_different(sentence_one, sentence_two, word_threshold=3):
    def tokenize(text):
        return re.findall(r"\w+", text.lower())

    tokens_one = tuple(tokenize(sentence_one))
    tokens_two = tuple(tokenize(sentence_two))

    @lru_cache(maxsize=None)
    def calculate_word_distance(index_one, index_two):
        if index_one == 0:
            return index_two
        if index_two == 0:
            return index_one

        if tokens_one[index_one - 1] == tokens_two[index_two - 1]:
            substitution_cost = 0
        else:
            substitution_cost = 1

        return min(
            calculate_word_distance(index_one - 1, index_two) + 1,
            calculate_word_distance(index_one, index_two - 1) + 1,
            calculate_word_distance(index_one - 1, index_two - 1) + substitution_cost,
        )

    total_distance = calculate_word_distance(len(tokens_one), len(tokens_two))
    return total_distance >= word_threshold


def paraphrase(
    sentence: str,
    api_key: str,
    system_prompt: str,
    index=None,
) -> str:
    if not is_natural_prose(sentence):
        return ""
    client = anthropic.Anthropic(api_key=api_key)
    if index:
        print(f"Paraphrasing Sentence #{index}")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": sentence}],
    )
    return message.content[0].text.strip()


def extract_paraphrases(
    paraphrases_xml_string: str,
    original_sentence: str,
    semantic_treshold: float,
    sentence_transformer,
    word_treshold: int,
) -> list[str]:
    parpahrases_inital = re.findall(
        r"<p\d>(.*?)</p\d>", paraphrases_xml_string, re.DOTALL
    )
    paraphrases_filtered = [
        paraphrase
        for paraphrase in parpahrases_inital
        if (
            is_semantically_similar(
                original_sentence, paraphrase, semantic_treshold, sentence_transformer
            )
            and is_sufficiently_different(original_sentence, paraphrase, word_treshold)
        )
    ]
    return (paraphrases_filtered + [None] * 4)[:4]


def safe_get(list, index):
    return list[index] if index < len(list) else None


def load_safely(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path '{file_path}' does not exist.")

    if not os.path.isfile(file_path):
        raise ValueError(f"The path '{file_path}' is a directory, not a file.")

    _, extension = os.path.splitext(file_path)
    if extension.lower() != ".tsv":
        raise ValueError(f"Invalid extension '{extension}'. Expected '.tsv'.")

    try:
        return pd.read_csv(file_path, sep="\t", on_bad_lines="error")
    except pd.errors.ParserError as e:
        raise ValueError(f"TSV Structure Error: {e}")


def save_paper_as_markdown(paper_text, storage_path):
    with open(storage_path, "w", encoding="utf-8") as f:
        f.write(paper_text)


def log_paper_results(
    csv_path, original_score, improved_score, paraphrases_path, paper_path
):
    new_data = pd.DataFrame(
        [
            {
                "paper": os.path.basename(paper_path),
                "paraphrases": os.path.basename(paraphrases_path),
                "original_score": original_score,
                "improved_score": improved_score,
            }
        ]
    )

    if not os.path.isfile(csv_path):
        new_data.to_csv(csv_path, index=False)
    else:
        new_data.to_csv(csv_path, mode="a", header=False, index=False)


def create_paraphrases(
    system_prompt_path: str,
    paper_string: str,
    anthropic_api_key: str,
    paraphrases_path: str,
    semantic_treshold: float,
    sentence_transformer,
    word_threshold: int,
) -> None:
    if paraphrases_path and os.path.isfile(paraphrases_path):
        return load_safely(file_path=paraphrases_path), False

    paper_sentences = extract_sentences(clean_markdown_latex(paper_string))
    system_prompt = load_file_as_string(system_prompt_path)
    paraphrases_lists = [
        extract_paraphrases(
            paraphrase(sentence, anthropic_api_key, system_prompt, index),
            sentence,
            semantic_treshold,
            sentence_transformer,
            word_threshold,
        )
        for index, sentence in enumerate(paper_sentences)
    ]
    paraphrases_dataframe = pd.DataFrame(
        {
            "originals": paper_sentences,
            **{
                f"paraphrase_{i+1}": [safe_get(p, i) for p in paraphrases_lists]
                for i in range(4)
            },
        }
    )
    paraphrases_dataframe.to_csv(paraphrases_path, sep="\t", index=False)
    return paraphrases_dataframe, True


import math


def logprobs_weighted_sum(output):
    try:
        content = output["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return 0.0

    allowed = {str(i): i for i in range(1, 11)}

    extracted = [
        (allowed[item["token"].strip()], math.exp(item["logprob"]))
        for item in content
        if item["token"].strip() in allowed
    ]

    if not extracted:
        text = output["choices"][0]["message"]["content"].strip()
        return float(text) if text.isdigit() else 1.0

    total_mass = sum(p for _, p in extracted)

    return sum(val * (p / total_mass) for val, p in extracted)


async def call_endpoint(
    messages,
    endpoint_url,
    headers,
    max_tokens,
    client,
    temperature=0.0,
    top_logprobs=None,
    guided_options=None,
):
    target_url = f"{endpoint_url.rstrip('/')}/v1/chat/completions"

    payload = {
        "model": "/data/Llama-OpenReviewer-8B",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if top_logprobs is not None:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs

    if guided_options:
        payload["guided_choice"] = guided_options

    response = await client.post(
        target_url, headers=headers, json=payload, timeout=400.0
    )
    response.raise_for_status()
    return response.json()


@time_it
async def evaluate_paper(
    paper_text,
    endpoint_url,
    headers,
    load_file_as_string,
    review_prompt_path,
    rating_prompt_path,
    client=None,
):
    if client is not None:
        return await _evaluate(
            paper_text,
            endpoint_url,
            headers,
            load_file_as_string,
            review_prompt_path,
            rating_prompt_path,
            client,
        )
    else:
        async with httpx.AsyncClient(
            timeout=400.0, follow_redirects=True
        ) as new_client:
            return await _evaluate(
                paper_text,
                endpoint_url,
                headers,
                load_file_as_string,
                review_prompt_path,
                rating_prompt_path,
                new_client,
            )


async def _evaluate(
    paper_text,
    endpoint_url,
    headers,
    load_file_as_string,
    review_prompt_path,
    rating_prompt_path,
    client,
):
    r_output = await call_endpoint(
        [
            {"role": "system", "content": load_file_as_string(review_prompt_path)},
            {"role": "user", "content": f"Please review:\n\n{paper_text}"},
        ],
        endpoint_url,
        headers,
        1000,
        client,
    )
    reasoning = r_output["choices"][0]["message"]["content"]

    return await call_endpoint(
        [
            {"role": "system", "content": load_file_as_string(rating_prompt_path)},
            {
                "role": "user",
                "content": f"Paper:\n{paper_text}\n\nReview:\n{reasoning}\n\nRating:",
            },
        ],
        endpoint_url,
        headers,
        1,
        client,
        0.0,
        20,
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    )
