import re
from copy import deepcopy
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import nltk
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from modern_slavery_registry.utils import CheckType

from . import utils

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


def find_special_chars(text: str, regular_chars: str = r"[^A-Za-z0-9 ]+"):
    """Find all special characters in given string.

    Args:
        text: A string

        regular_chars: regular expression for regular/normal characters

    Returns:
        A list of special characters extracted from given string
    """
    return list(set(re.findall(regular_chars, text)))


def replace_special_chars(
    text: str, regular_chars: str = r"[^A-Za-z0-9 ]+", replace_with: str = " "
) -> str:
    """Replace special characters from text."""
    return re.sub(regular_chars, replace_with, text)


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
    tokens: List[Union[str, List[str]]],
    text: str,
    splitter: str = ". ",
) -> Tuple[List[str], str]:
    """Remove sentences with any of tokens in given text.

    Args:
        tokens: sentences with any of tokens are removed

        text: consists of one or more sentences

        splitter: use to split text into list of sentences

    Returns:
        text with sentences without any of given tokens
    """
    sentences = text.split(splitter)
    n_sentences = len(sentences)

    if not isinstance(tokens, list):
        raise ValueError(
            f"Expected tokens to be of type list, got {type(tokens)}"
        )

    def are_all_tokens_in_sentence(tokens: List[str], sentence: str) -> bool:
        all_tokens_in_sentence = True
        for token in tokens:
            if len(re.findall(fr"\b{token}\b", sentence)) == 0:
                all_tokens_in_sentence = False
                break
        return all_tokens_in_sentence

    keep_sentences = []
    for sentence in sentences:
        any_sub_tokens_in_sentence = False
        for sub_tokens in tokens:
            if isinstance(sub_tokens, str):
                sub_tokens = [sub_tokens]
            if are_all_tokens_in_sentence(sub_tokens, sentence):
                any_sub_tokens_in_sentence = True
                break
        if not any_sub_tokens_in_sentence:
            keep_sentences.append(sentence)

    if len(keep_sentences) == 1:
        keep_sentences.append("")

    return f"{splitter}".join(keep_sentences)


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


eng_stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


def remove_stopwords(text: str, to_lower: bool = False) -> str:
    """Remove stopwords from text."""
    if to_lower:
        text = text.lower()
    return " ".join(
        [word for word in text.split() if word not in eng_stopwords]
    )


def nltk_tag_to_wordnet_tag(nltk_tag: str) -> Union[str, None]:
    """Convert NLTK tag to WORDNET tag.

    Args:
        nltk_tag: string

    Returns:
        A wordnet tag if found else None
    """
    if nltk_tag.startswith("J"):
        return wordnet.ADJ
    elif nltk_tag.startswith("V"):
        return wordnet.VERB
    elif nltk_tag.startswith("N"):
        return wordnet.NOUN
    elif nltk_tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence: str) -> str:
    """Lemmatize words in given sentence.

    Args:
        sentence: string

    Note:
        Requires "punkt", "averaged_perceptron_tagger" and "wordnet" nltk resources

    Returns:
        A sentence with lemmatized words
    """
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = [
        (token, nltk_tag_to_wordnet_tag(nltk_tag))
        for token, nltk_tag in nltk_tagged
    ]
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def _compute_ngram_freqs(text: str, n: int = 1) -> FreqDist:
    """Compute frequency of ngrams in given sentence.

    Args:
        text: A string

        n: ngram

    Returns:
        NLTK frequency distribution object with frequencies corresponding to ngrams.

    Usage:
        >>> freq = text_parser.compute_ngram_freq("this is a sentence", 1)
        >>> freq.most_common()
        [(('this',), 1), (('is',), 1), (('a',), 1), (('sentence',), 1)]
    """
    tokens = nltk.word_tokenize(text)
    ngrams = nltk.ngrams(sequence=tokens, n=n)
    return nltk.FreqDist(samples=ngrams)


def compute_ngram_freqs(corpus: List[str], n: int = 1, verbose: bool = False):
    """Compute ngram fequencies in given corpus.

    Args:
        corpus: List of strings

        n: ngram

    Returns:
        A dictionary with words and their corresponding frequencies.
    """
    CheckType((list, tuple, np.ndarray), corpus, "corpus")

    freqs = {}
    if verbose:
        corpus = tqdm(corpus, leave=False, position=0)
    for text in corpus:
        freq_dist = _compute_ngram_freqs(text, n)
        for token, freq in freq_dist.most_common():
            if token in freqs:
                freqs[token] += freq
            else:
                freqs[token] = freq
    return freqs
