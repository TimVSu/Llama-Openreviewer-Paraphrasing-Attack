def clean_markdown_latex(md_string: str, reduce, re) -> str:
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


def extract_sentences(cleaned_text: str, nltk) -> list[str]:
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


def paraphrase(sentence: str, api_key: str, system_prompt: str, anthropic) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": sentence}],
    )
    return message.content[0].text.strip()


def extract_paraphrases(paraphrases_xml_string: str) -> list[str]:
    return re.findall(r"<p\d>(.*?)</p\d>", paraphrases_xml_string, re.DOTALL)


def safe_get(lst, i):
    return lst[i] if i < len(lst) else None


def load_safely(file_path: str, pandas):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path '{file_path}' does not exist.")

    if not os.path.isfile(file_path):
        raise ValueError(f"The path '{file_path}' is a directory, not a file.")

    _, extension = os.path.splitext(file_path)
    if extension.lower() != ".tsv":
        raise ValueError(f"Invalid extension '{extension}'. Expected '.tsv'.")

    try:
        return pandas.read_csv(file_path, sep="\t", on_bad_lines="error")
    except pandas.errors.ParserError as e:
        raise ValueError(f"TSV Structure Error: {e}")


def create_paraphrases(
    system_prompt_path: str,
    paper_string: str,
    anthropic_api_key: str,
    paraphrases_path: str,
    reduce,
    re,
    pandas,
) -> None:
    if not os.path.isfile(paraphrases_path):
        paper_sentences = extract_sentences(
            clean_markdown_latex(paper_string, reduce, re)
        )
        system_prompt = load_file_as_string(system_prompt_path)
        paraphrases_lists = [
            extract_paraphrases(paraphrase(sentence, ANTHROPIC_API_KEY, system_prompt))
            for sentence in paper_sentences
        ]
        paraphrases_dataframe = pandas.Dataframe(
            {
                "originals": paper_sentences,
                **{
                    f"paraphrase_{i+1}": [safe_get(p, i) for p in paraphrases_lists]
                    for i in range(4)
                },
            }
        )
        paraphrases_dataframe.to_csv(paraphrases_path, sep="\t", index=False)
        return paraphrases_dataframe
    return load_safely(file_path=paraphrases_path, pandas=pandas)

def evaluate_paper(eval_prompt_path: str, huggingface):
