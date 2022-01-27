import os
import re
from typing import Dict, List, Optional

import pandas as pd
from urlextract import URLExtract

from modern_slavery_registry import text_parser, utils

DATA_PATH = "/home/mittal.nit/projects/modern_slavery_registry/data"
uk_to_us_mappings = pd.read_excel(
    os.path.join(DATA_PATH, "UK_to_US_word_mappings.xlsx")
)
uk_to_us_mappings = {row[0]: row[1] for row in uk_to_us_mappings.values}

url_extractor = URLExtract()


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    urls = url_extractor.find_urls(text)
    for url in urls:
        text = text.replace(url, " ")

    remove_tokens = [
        ["privacy", "policy"],
        "cookies",
        "cookie",
        ["skip", "content"],
        ["please", "donate"],
        ["service", "inquiry"],
        ["contact", "page"],
        "signup",
        "download",
        "newsletter",
    ]
    text = text_parser.remove_sentences_with_tokens(
        tokens=remove_tokens, text=text
    )

    text = text_parser.remove_stopwords(text)

    words_in_text = []
    for word in text.split():
        word = (
            text_parser.word_expantions[word]
            if word in text_parser.word_expantions
            else word
        )
        word = uk_to_us_mappings[word] if word in uk_to_us_mappings else word
        words_in_text.append(word)

    text = " ".join(words_in_text)

    text = text_parser.replace_special_chars(
        text, regular_chars=r"[^A-Za-z0-9.,:;?$ ]+", replace_with=""
    )

    # additional filtering and removing phrases
    remove_phrases = [
        r"\buk modern slavery act statement\b",
        r"\buk modern slavery act\b",
        r"\bcalifornia transparency supply chain act statement\b",
        r"\bcalifornia transparency supply chain act\b",
        r"\bcalifornia transparency supply chain\b",
        r"\bcalifornia transparency supply\b",
        r"\bcalifornia transparency\b",
        r"\bmodern slavery act statement\b",
        r"\btransparency supply chain act statement\b",
        r"\btransparency supply chain act\b",
        r"\bmodern slavery statement\b",
        r"\bmodern slavery act\b",
        r"\bmodern slavery\b",
        r"\bmodern day slavery statement\b",
        r"\bmodern day slavery\b",
        r"\bslavery\b",
        r"\bsupply chain\b",
        r"\bhuman trafficking statement\b",
        r"\bpolicy statement\b",
        r"\btrafficking\b",
        r"\bltd\b",
        r"\b\w\b",
    ]

    for phrase in remove_phrases:
        text = re.sub(phrase, " ", text)

    text = " ".join(text.split())

    return text_parser.lemmatize_sentence(text)


def clean_corpus(
    corpus: List[str], max_workers: Optional[int] = None
) -> Dict[int, str]:
    import concurrent.futures

    cleaned_corpus = dict()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = {
            executor.submit(clean_text, text): i
            for i, text in enumerate(corpus)
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures), leave=False, position=0
        ):
            i = futures[future]
            try:
                cleaned_corpus[i] = future.result()
            except Exception as e:
                print(f"Failed for statement no.: {i}, Exception: {e}")

    return cleaned_corpus
