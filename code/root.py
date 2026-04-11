import pandas
import os
import json
import asyncio
import httpx
from dotenv import load_dotenv
from utils import (
    load_file_as_string,
    create_paraphrases,
    evaluate_paper,
    logprobs_weighted_sum,
    save_paper_as_markdown,
    log_paper_results,
    call_endpoint,
)
from sentence_transformers import SentenceTransformer, util

load_dotenv(".env.local")
load_dotenv(".env")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
PAPER_PATH = os.environ.get("PAPER_PATH")
PARAPHRASING_PROMPT_PATH = os.environ.get("PARAPHRASING_PROMPT_PATH")
SEMANTIC_THRESHOLD = float(os.environ.get("SEMANTIC_THRESHOLD"))
TIMEOUT = float(os.environ.get("TIMEOUT"))
WORD_THRESHOLD = int(os.environ.get("WORD_THRESHOLD"))
INFERENCE_HEADERS = {
    "Authorization": f"Bearer {os.environ.get('INFERENCE_TOKEN')}",
    "Content-Type": "application/json",
}
INFERENCE_ENDPOINT_URL = os.environ.get("INFERENCE_ENDPOINT_URL")
INFERENCE_ENDPOINT_MODEL = os.environ.get("INFERENCE_ENDPOINT_MODEL")
RESULTS_PATH = os.environ.get("RESULTS_PATH")
SCORES_PATH = os.environ.get("SCORES_PATH")
REVIEW_PROMPT_PATH = os.environ.get("REVIEW_PROMPT_PATH")
RATING_PROMPT_PATH = os.environ.get("RATING_PROMPT_PATH")

FINAL_REVIEWS_DIR = os.environ.get("FINAL_REVIEWS_DIR", "./final_reviews")
FINAL_EVAL_PROMPT_PATH = os.environ.get("FINAL_EVAL_PROMPT_PATH")

PARAPHRASES_PATH = os.path.join(
    os.environ.get("PARAPHRASES_DIR"),
    f"{os.path.splitext(os.path.basename(PAPER_PATH))[0]}_paraphrases_{int(SEMANTIC_THRESHOLD * 100)}_{WORD_THRESHOLD}.tsv",
)

CHECKPOINT_PATH = os.path.join(
    os.environ.get("CHECKPOINT_DIR"),
    f"{os.path.splitext(os.path.basename(PAPER_PATH))[0]}.json",
)

sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")


async def optimize_paper(df, paper_text, review_path, rating_path):
    optimized_text = paper_text
    start_index = 0
    original_score = None

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            checkpoint = json.load(f)
            optimized_text = checkpoint["text"]
            current_score = checkpoint["score"]
            start_index = checkpoint["index"] + 1
            original_score = checkpoint.get("original_score")
            print(
                f"Resuming from index {start_index}. Original score was: {original_score}"
            )
    else:
        print("Calculating original score...")
        initial_output = await evaluate_paper(
            paper_text,
            INFERENCE_ENDPOINT_URL,
            INFERENCE_HEADERS,
            load_file_as_string,
            review_path,
            rating_path,
            timeout=TIMEOUT,
            model=INFERENCE_ENDPOINT_MODEL,
        )
        original_score = logprobs_weighted_sum(initial_output)
        current_score = original_score
        print(f"Original score: {original_score}")

    sem = asyncio.Semaphore(5)

    async def limited_evaluate(opt_text, client):
        async with sem:
            return await evaluate_paper(
                opt_text,
                INFERENCE_ENDPOINT_URL,
                INFERENCE_HEADERS,
                load_file_as_string,
                review_path,
                rating_path,
                timeout=TIMEOUT,
                model=INFERENCE_ENDPOINT_MODEL,
                client=client,
            )

    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
        for index, row in df.iloc[start_index:].iterrows():
            print(f"Optimizing sentence #{index}")
            original_phrase = row["originals"]
            paraphrase_options = [
                row[f"paraphrase_{i+1}"]
                for i in range(4)
                if pandas.notna(row[f"paraphrase_{i+1}"])
            ]

            tasks = [
                limited_evaluate(optimized_text.replace(original_phrase, opt), client)
                for opt in paraphrase_options
            ]

            results = await asyncio.gather(*tasks)

            for i, output in enumerate(results):
                new_score = logprobs_weighted_sum(output)
                if new_score > current_score:
                    print(f"New score is: {new_score}")
                    print(f"original sentence is: {original_phrase}")
                    print(f"sentence was replaced by: {paraphrase_options[i]}")
                    current_score = new_score
                    optimized_text = optimized_text.replace(
                        original_phrase, paraphrase_options[i]
                    )

            with open(CHECKPOINT_PATH, "w") as f:
                json.dump(
                    {
                        "index": index,
                        "score": current_score,
                        "text": optimized_text,
                        "original_score": original_score,
                    },
                    f,
                )

    return optimized_text, current_score, original_score


async def generate_final_review(content, label, client):
    system_content = load_file_as_string(FINAL_EVAL_PROMPT_PATH)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Review the following paper:\n\n{content}"},
    ]

    paper_name = os.path.splitext(os.path.basename(PAPER_PATH))[0]
    threshold_val = int(SEMANTIC_THRESHOLD * 100)
    filename = f"{paper_name}_{label}_{threshold_val}_{WORD_THRESHOLD}.txt"
    output_path = os.path.join(FINAL_REVIEWS_DIR, filename)

    print(f"Generating final {label} review...")
    response_json = await call_endpoint(
        messages=messages,
        endpoint_url=INFERENCE_ENDPOINT_URL,
        headers=INFERENCE_HEADERS,
        max_tokens=1500,
        client=client,
        temperature=0.0,
    )
    review_text = response_json["choices"][0]["message"]["content"]

    os.makedirs(FINAL_REVIEWS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(review_text)
    print(f"Review saved to: {output_path}")


async def main():
    paper_text = load_file_as_string(PAPER_PATH)
    paraphrases, isNew = create_paraphrases(
        PARAPHRASING_PROMPT_PATH,
        paper_text,
        ANTHROPIC_API_KEY,
        PARAPHRASES_PATH,
        SEMANTIC_THRESHOLD,
        sentence_transformer,
        WORD_THRESHOLD,
    )

    if isNew:
        user_input = (
            input("Paraphrases created. Run experiment? (y/n): ").lower().strip()
        )
        if user_input == "n":
            exit()

    final_text, final_score, original_score = await optimize_paper(
        paraphrases, paper_text, REVIEW_PROMPT_PATH, RATING_PROMPT_PATH
    )

    print(f"Optimization complete.\nOriginal: {original_score}\nFinal: {final_score}")

    save_paper_as_markdown(final_text, RESULTS_PATH)
    log_paper_results(
        SCORES_PATH,
        original_score=original_score,
        improved_score=final_score,
        paraphrases_path=PARAPHRASES_PATH,
        paper_path=PAPER_PATH,
    )

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        await generate_final_review(paper_text, "original", client)
        await generate_final_review(final_text, "improved", client)

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


if __name__ == "__main__":
    asyncio.run(main())
