import re
import nltk
from nltk.corpus import stopwords
from typing import List, Union, Dict, Tuple, Sequence

eng_stopwords = stopwords.words('english')

def find_urls_in_text(text : str) -> List:
    """Return list of urls extracted from text."""
    # findall() has been used  
    # with valid conditions for urls in string 
    regex = (r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)"
             "(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()"
             "<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))")
    url = re.findall(regex, text)

    # x[0] because different types of urls will be caught and non-empty type url be at index 0 for every url type caught
    # ex. [('https://www.wsj.com/articles/todays-top-supply-chain-and-logistics-news-from-wsj-1462876696',
    #       '',
    #       '',
    #       '',
    #       ''),
    #      ('http://on.wsj.com/Logisticsnewsletter', '', '', '', '')]
    return [x[0] for x in url]


def replace_urls(text: str, 
                 replace_with: str= ""):
    urls = find_urls_in_text(text)
    for url in urls:
        text = text.replace(url, " ")
    return text
    
    
def replace_unicode(text: str, 
                    replace_with: str=" ") -> str:
    return re.sub(r'[^\x00-\x7F]+',replace_with, text)


def replace_special_chars(text, 
                          replace_digits:bool = False,
                          replace_with: str=" ") -> str:
    if replace_digits:
        pattern = r"[^A-Za-z]+"
    else:
        pattern = r"[^A-Za-z0-9]+"
    return re.sub(pattern, replace_with, text)


def remove_stopwords(text:str):
    text = text.lower()
    return " ".join([word for word in text.split() if word not in eng_stopwords])


def find_ngrams_in_text(sentence: str,
                        text: str, 
                        to_lower: bool= True) -> Dict:
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
    for i in range(1, len(sentence)+1):
        name = sentence[:i]
        mapping[" ".join(name)] = len(re.findall(r"\b" + " ".join(name) + r"\b", text))
    return mapping