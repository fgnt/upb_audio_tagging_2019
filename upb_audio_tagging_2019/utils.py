
import collections
import numpy as np
import torch
import datetime


def to_list(x, length=None):
    """
    Often a list is required, but for convenience it is desired to enter a
    single object, e.g. string.

    Complicated corner cases are e.g. `range()` and `dict.values()`, which are
    handled here.

    >>> to_list(1)
    [1]
    >>> to_list([1])
    [1]
    >>> to_list((i for i in range(3)))
    [0, 1, 2]
    >>> to_list(np.arange(3))
    [0, 1, 2]
    >>> to_list({'a': 1})
    [{'a': 1}]
    >>> to_list({'a': 1}.keys())
    ['a']
    >>> to_list('ab')
    ['ab']
    >>> from pathlib import Path
    >>> to_list(Path('/foo/bar'))
    [PosixPath('/foo/bar')]
    """
    # Important cases (change type):
    #  - generator -> list
    #  - dict_keys -> list
    #  - dict_values -> list
    #  - np.array -> list (discussable)
    # Important cases (list of original object):
    #  - dict -> list of dict

    def to_list_helper(x_):
        return [x_] * (1 if length is None else length)

    if isinstance(x, collections.Mapping):
        x = to_list_helper(x)
    elif isinstance(x, str):
        x = to_list_helper(x)
    elif isinstance(x, collections.Sequence):
        pass
    elif isinstance(x, collections.Iterable):
        x = list(x)
    else:
        x = to_list_helper(x)

    if length is not None:
        assert len(x) == length, (len(x), length)
    return x


def timestamp(fmt='%Y-%m-%d-%H-%M-%S'):
    return datetime.datetime.now().strftime(fmt)
