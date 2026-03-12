import pandas
import nltk
import os
import anthropic
from functools import reduce
import re
from utils import load_file_as_string, create_paraphrases


ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
PAPER_PATH = os.environ.get("PAPER_PATH")
PARAPHRASING_PROMPT_PATH = os.environ.get("PARAPHRASING_PROMPT_PATH")
PARAPHRASES_PATH = os.environ.get("PARAPHRASES_PATH")
EVAL_PROMPT_PATH = os.environ.get("EVAL_PROMPT_PATH")

paper_text = load_file_as_string(PAPER_PATH)
paraphrases = create_paraphrases(
    system_prompt=PARAPHRASING_PROMPT_PATH,
    paper_string=paper_text,
    anthropic_api_key=ANTHROPIC_API_KEY,
    paraphrases_path=PARAPHRASES_PATH,
    reduce=reduce,
    re=re,
    pandas=pandas,
)
