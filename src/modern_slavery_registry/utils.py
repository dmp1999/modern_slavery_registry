# import multiprocessing.pool
import functools
import os
from pickle import dump, load
from typing import Any, Dict

import numpy as np
import pandas as pd


def fix_unicode(df: pd.core.frame.DataFrame):
    return df.applymap(
        lambda x: x.encode("unicode_escape").decode("utf-8")
        if isinstance(x, str)
        else x
    )


def sort_dict(dict_: Dict, by: int = 0, reverse: bool = False):
    """Sort and return input dictionary"""
    return {
        k: v for k, v in sorted(dict_.items(), key=lambda x: x[by], reverse=reverse)
    }


def count_num_in_between(arr, min_, max_):
    return np.sum((min_ <= arr) & (arr <= max_))


def prepare_path(filename: str, path: str = None) -> str:
    format = ".pickle"
    if len(filename) <= len(format) or filename[len(format)] != format:
        filename += format
    if path is not None:
        filename = os.path.join(path, filename)
    return filename


def dump_pickle(obj: Any, filename: str, path: str = None):
    filename = prepare_path(filename, path)
    with open(file=filename, mode="wb") as f:
        dump(obj, f)


def load_pickle(filename: str, path: str = None) -> Any:
    filename = prepare_path(filename, path)
    with open(file=filename, mode="rb") as f:
        return load(f)
