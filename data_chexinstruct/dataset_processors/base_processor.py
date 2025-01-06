import time
from types import FunctionType


class BaseProcessor:
    instruct = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr, value in cls.__dict__.items():
            if isinstance(value, FunctionType):
                setattr(cls, attr, cls.timeit(value))

    @staticmethod
    def timeit(method):
        def timed(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            print(f"{args[0].__class__.__name__}.{method.__name__} took: {te - ts:.2f} sec")
            return result

        return timed
