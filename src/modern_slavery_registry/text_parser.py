import re
from copy import deepcopy
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from . import utils

eng_stopwords = stopwords.words("english")

# list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python """
word_expantions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
}


def find_urls_in_text(text: str) -> List:
    """Return list of urls extracted from text."""
    # findall() has been used
    # with valid conditions for urls in string
    regex = (
        r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)"
        "(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()"
        "<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    )
    url = re.findall(regex, text)

    # x[0] because different types of urls will be caught and non-empty type url be at index 0 for every url type caught
    # ex. [('https://www.wsj.com/articles/todays-top-supply-chain-and-logistics-news-from-wsj-1462876696',
    #       '',
    #       '',
    #       '',
    #       ''),
    #      ('http://on.wsj.com/Logisticsnewsletter', '', '', '', '')]
    return [x[0] for x in url]


def replace_urls(text: str, replace_with: str = ""):
    """Replace URLs from text."""
    urls = find_urls_in_text(text)
    for url in urls:
        text = text.replace(url, " ")
    return text


def replace_unicode(text: str, replace_with: str = " ") -> str:
    """Replace Unicode from text."""
    return re.sub(r"[^\x00-\x7F]+", replace_with, text)


def replace_special_chars(
    text, replace_digits: bool = False, replace_with: str = " "
) -> str:
    """Replace special characters from text."""
    if replace_digits:
        pattern = r"[^A-Za-z]+"
    else:
        pattern = r"[^A-Za-z0-9]+"
    return re.sub(pattern, replace_with, text)


def remove_stopwords(text: str):
    """Remove stopwords from text."""
    text = text.lower()
    return " ".join(
        [word for word in text.split() if word not in eng_stopwords]
    )


def find_ngrams_in_text(
    sentence: str, text: str, to_lower: bool = True
) -> Dict[str, int]:
    """
    Find number of instances of sentence ngrams in text.

    Sentence is converted into ngrams of words

    Parameters
    ----------
    sentence: str

    text: str

    Returns
    -------
    mapping: Dict, {str:int}

    Examples
    --------
    >>> find_ngrams_in_text("General Motors",
        "General Motors Company(GM) is an American "
        "multinational corporation headquartered in Detroit. "
        "General Motors manufactures vehicles in several countries.")
    {"general": 2, "general motors": 2}
    """
    if to_lower:
        sentence = sentence.lower()
        text = text.lower()
    sentence = sentence.split()
    mapping = {}
    for i in range(1, len(sentence) + 1):
        name = sentence[:i]
        mapping[" ".join(name)] = len(
            re.findall(r"\b" + " ".join(name) + r"\b", text)
        )
    return mapping


def remove_sentences_with_tokens(
    text: str, tokens: List[Union[str, List[str]]], split_at: str = ". "
) -> Tuple[List[str], str]:
    """
    Remove all sentences with input tokens.

    # Note: can use multithreading or parallel processing
    """
    sentences = text.split(split_at)
    n_sentences = len(sentences)

    def remove_sentence_with_token(
        sentences: List[str], token: List[str], i: int
    ):
        if i == n_sentences:
            return sentences
        for sentence in sentences:
            is_token_present_in_sentence = True
            for sub_token in token:
                if len(list(re.finditer(fr"\b{sub_token}\b", sentence))) == 0:
                    # if sentence.find(sub_token) == -1:
                    is_token_present_in_sentence = False
            if is_token_present_in_sentence:
                sentences.remove(sentence)
                return remove_sentence_with_token(
                    deepcopy(sentences), token, i + 1
                )
        return sentences

    for token in tokens:
        if isinstance(token, str):
            token = [token]
        sentences = remove_sentence_with_token(deepcopy(sentences), token, 0)

    return f"{split_at}".join(sentences)


def generate_vocab(texts: Union[str, Iterable[str]]) -> Dict[str, int]:
    """Generate vocab for input corpus."""
    vocab = {}

    def fill_vocab(text: str, vocab: Dict[str, int]) -> Dict[str, int]:
        for word in text.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
        return vocab

    if isinstance(texts, str):
        return fill_vocab(texts, vocab)
    else:
        for text in texts:
            fill_vocab(text, vocab)

    return vocab


def generate_ngrams(
    corpus: List[str], ngrams: Tuple[int, int] = (1, 1)
) -> List[List[str]]:
    """Generate ngrams from input list of sentences."""
    ngrams_from_corpus = []
    for sentence in tqdm(corpus, position=0, leave=True):
        sentence = sentence.split()
        ngrams_from_sentence = []
        len_sentence = len(sentence)
        for n in range(ngrams[0], ngrams[1] + 1):
            for i in range(len_sentence - n + 1):
                ngrams_from_sentence.append(" ".join(sentence[i : i + n]))
        #     # preparing ngrams at end of sentence
        #     for i in range(len_sentence-ngram+1, len_sentence):
        #         ngram_sentence.append(" ".join(
        #             sentence[i:] + ["$PAD$"] * (ngram -  len(sentence[i :]))))
        ngrams_from_corpus.append(ngrams_from_sentence)
    return ngrams_from_corpus


def compute_term_doc_freq(
    ngrams_from_corpus: List[List[str]],
    sort: bool = False,
    descending: bool = False,
) -> Union[Dict[str, int], Dict[str, int]]:
    """Compute term and document frequency from ngrams."""
    term_freq = {}  # to keep track of term frequency
    doc_freq = {}  # to keep track of document-term frequency
    ngram_last_doc = {}
    n = len(ngrams_from_corpus)
    for i, ngrams_from_sentence in tqdm(enumerate(ngrams_from_corpus)):
        for ngram in ngrams_from_sentence:
            if ngram not in term_freq:
                term_freq[ngram] = 1
                doc_freq[ngram] = 1
            else:
                term_freq[ngram] += 1
                if ngram_last_doc[ngram] != i:
                    doc_freq[ngram] += 1
            ngram_last_doc[ngram] = i

    if sort:
        term_freq = utils.sort_dict(dict_=term_freq, by=1, reverse=descending)
        doc_freq = utils.sort_dict(dict_=doc_freq, by=1, reverse=descending)
    return term_freq, doc_freq
