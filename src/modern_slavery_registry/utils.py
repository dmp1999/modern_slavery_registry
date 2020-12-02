import pandas as pd
import multiprocessing.pool
import functools

def fix_unicode(df: pd.core.frame.DataFrame):
    return df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

# https://stackoverflow.com/a/35139284
def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(max_timeout)
        return func_wrapper
    return timeout_decorator