import asyncio
import httpx
import os
from dotenv import load_dotenv
from utils import call_endpoint, load_file_as_string, save_paper_as_markdown

load_dotenv(".env.local")
load_dotenv(".env")


async def run_final_check():
    ENDPOINT_URL = os.environ.get("INFERENCE_ENDPOINT_URL")
    HEADERS = {}
    FINAL_EVAL_PROMPT_PATH = "../prompts/final_review_prompt.txt"
    PAPER_PATH = "../results/paper/435_improved.md"
    OUTPUT_PATH = "./435_improved_result.txt"

    system_content = load_file_as_string(FINAL_EVAL_PROMPT_PATH)
    paper_content = load_file_as_string(PAPER_PATH)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Review the following paper:\n\n{paper_content}"},
    ]

    async with httpx.AsyncClient(timeout=400.0) as client:
        try:
            print("Starting final check...")
            response_json = await call_endpoint(
                messages=messages,
                endpoint_url=ENDPOINT_URL,
                headers=HEADERS,
                max_tokens=1500,
                client=client,
                temperature=0.0,
            )

            final_result = response_json["choices"][0]["message"]["content"]

            save_paper_as_markdown(final_result, OUTPUT_PATH)
            print(f"Final check complete. Result saved to: {OUTPUT_PATH}")

        except Exception as e:
            print(f"An error occurred during the final check: {e}")


if __name__ == "__main__":
    asyncio.run(run_final_check())


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
                client,
            )

    async with httpx.AsyncClient(timeout=400.0, follow_redirects=True) as client:
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
                    print(f"sentece was replaced by: {paraphrase_options[i]}")
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
