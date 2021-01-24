# import multiprocessing.pool
import functools
import os
from hashlib import md5
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


# # https://stackoverflow.com/a/35139284
# def timeout(max_timeout):
#     """Timeout decorator, parameter in seconds."""
#     def timeout_decorator(item):
#         """Wrap the original function."""
#         @functools.wraps(item)
#         def func_wrapper(*args, **kwargs):
#             """Closure for function."""
#             pool = multiprocessing.pool.ThreadPool(processes=1)
#             async_result = pool.apply_async(item, args, kwargs)
#             # raises a TimeoutError if execution exceeds max_timeout
#             return async_result.get(max_timeout)
#         return func_wrapper
#     return timeout_decorator


def generate_hash(string: str) -> str:
    """Generate hash for input string"""
    return md5(string=f"{string}".encode()).hexdigest()


def sort_dict(dict_: Dict, by: int = 0, reverse: bool = False):
    """Sort and return input dictionary"""
    return {
        k: v
        for k, v in sorted(dict_.items(), key=lambda x: x[by], reverse=reverse)
    }


def count_num_in_between(arr, min_, max_):
    return np.sum((min_ <= arr) & (arr <= max_))




def prepare_path(filename: str, path: str=None)->str:
  format = ".pickle"
  if len(filename) <= len(format) or filename[len(format)] != format:
    filename+=format
  if path is not None:
    filename=os.path.join(path, filename)
  return filename

def dump_pickle(obj: Any, filename: str, path:str=None):
  filename=prepare_path(filename, path)
  with open(file=filename, mode='wb') as f:
    dump(obj, f)

def load_pickle(filename: str, path: str=None)->Any:
  filename=prepare_path(filename, path)
  with open(file=filename, mode='rb') as f:
    return load(f)
