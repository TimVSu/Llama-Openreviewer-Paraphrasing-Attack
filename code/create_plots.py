import re
import sys
import matplotlib.pyplot as plt


def parse_log(log_text):
    """
    Parse a paraphrasing attack log and return a list of (sentence_index, score) pairs.
    Each sentence carries the most recent score up to the next score update.
    """
    lines = log_text.strip().splitlines()
    score_re = re.compile(r"New score is:\s*([\d.]+)")
    original_score_re = re.compile(r"Original score:\s*([\d.]+)")
    sentence_re = re.compile(r"Optimizing sentence #(\d+)")
    current_score = None
    pending_sentences = []
    results = {}  # sentence_index -> score (last update wins)
    original_score = None

    for line in lines:
        orig_match = original_score_re.search(line)
        if orig_match:
            original_score = float(orig_match.group(1))
            current_score = original_score
            continue
        sent_match = sentence_re.search(line)
        if sent_match:
            pending_sentences.append(int(sent_match.group(1)))
            continue
        score_match = score_re.search(line)
        if score_match:
            new_score = float(score_match.group(1))
            for idx in pending_sentences:
                results[idx] = new_score
            pending_sentences = []
            current_score = new_score
            continue

    for idx in pending_sentences:
        if current_score is not None:
            results[idx] = current_score

    return original_score, list(results.items())


def plot_scores(original_score, results, output_path=None):
    if not results:
        print("No sentence/score pairs found.")
        return

    sentences, scores = zip(*sorted(results))

    fig, ax = plt.subplots(figsize=(12, 5))

    first_idx = min(sentences)
    plot_x = [first_idx - 1] + list(sentences)
    plot_y = (
        [original_score] + list(scores) if original_score is not None else list(scores)
    )

    ax.step(plot_x, plot_y, where="post", color="#3266ad", linewidth=2)

    ax.set_xlabel("Sentence index", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Score progression during paraphrasing attack", fontsize=13)

    max_sentence = max(sentences)
    tick_marks = list(range(0, max_sentence + 1, 25))
    if max_sentence not in tick_marks:
        tick_marks.append(max_sentence)
    ax.set_xticks(tick_marks)

    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_attack_log.py <logfile> [output_image.png]")
        print("       cat logfile | python parse_attack_log.py -")
        sys.exit(1)

    log_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if log_path == "-":
        log_text = sys.stdin.read()
    else:
        with open(log_path, "r") as f:
            log_text = f.read()

    original_score, results = parse_log(log_text)

    print(f"Original score : {original_score}")
    print(f"\n{'Sentence':>10}  {'Score':>12}")
    print("-" * 26)
    for idx, score in sorted(results):
        marker = " ↑" if (original_score and score > original_score) else ""
        print(f"{idx:>10}  {score:>12.6f}{marker}")

    plot_scores(original_score, results, output_path)


if __name__ == "__main__":
    main()
