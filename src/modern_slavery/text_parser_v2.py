import re

# from copy import deepcopy
from typing import List, Union
from nltk.corpus import wordnet

import nltk

from abc import ABC

# import numpy as np


# from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer

# from tqdm import tqdm

# from modern_slavery_registry.utils import CheckType

# from . import utils

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


def find_urls_in_text(text: str) -> List[str]:
    """Return list of urls extracted from text.

    Args:
        text : string text

    Returns:
        A list of urls (if present) extracted in given text
    """
    # findall() has been used
    # with valid conditions for urls in string
    regex = (
        r"\b(([\w-]+://?|www[.])[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/)))"
    )
    urls = re.findall(regex, text)
    return [url[0] for url in urls]


class CustomWordNetLemmatizer(ABC):
    """Custom WordNet Lemmatizer.

    Note : NLTK pos tags are converted into WORDNET pos tags
        before lemmatization.
    """

    REQUIRED_RESOURCES = {
        "tokenizers/punkt": "punkt",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    def __init__(
        self,
    ):
        for (
            source,
            package,
        ) in CustomWordNetLemmatizer.REQUIRED_RESOURCES.items():
            try:
                nltk.data.find(source)
            except Exception:
                nltk.download(package)
        self.lemmatizer = WordNetLemmatizer()

    def nltk_tag_to_wordnet_tag(self, tag: str) -> Union[str, None]:
        """Convert NLTK tag to WORDNET tag.

        Args:
            tag : nltk string tag

        Returns:
            A wordnet tag for corresponding nltk tag
        """
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return None

    def lemmatize(self, text: str) -> str:
        """Lemmatize words in given text.

        Args:
            text : string

        Note : Require pre-downloaded "punkt", "averaged_perceptron_tagger"
            and "wordnet" from nltk

        Returns:
            Lemmatized text
        """
        # tokenize the text and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(text))
        # tuple of (token, wordnet_tag)
        wordnet_tagged = [
            (token, self.nltk_tag_to_wordnet_tag(nltk_tag))
            for token, nltk_tag in nltk_tagged
        ]
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(
                    self.lemmatizer.lemmatize(word, tag)
                )
        return " ".join(lemmatized_sentence)
