import os
import pickle

from shared.utils.decorators import Alias, aliased
from common.utils.util_func import create_dir
from common.paths import Paths


@aliased
class DataStore(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(s, kwargs: dict = None, path_buffer=None):
        s.path_buffer = path_buffer or Paths.path_buffer
        create_dir(s.path_buffer)
        if kwargs:
            for name, item in kwargs.items():
                s.__setattr__(name, item)

    def temp_save(s, path, fn):
        try:
            if type(s[fn]) != str:
                path_fn = os.path.join(path, fn)
                pickle.dump(s[fn], open(path_fn, 'wb'), protocol=4)
                s[fn] = path_fn
                return s
            else:
                return s
        except KeyError:
            return s

    @Alias('load', 'load_from_path')
    def d_load_path(s, el):
        try:
            try:
                if type(s[el]) is str:
                    with open(s[el], "rb") as f:
                        s[el] = pickle.load(f)
                    return s
                else:
                    return s
            except FileNotFoundError:
                print('Warning: File {} was not found. Check whether this file load is necessary at this point. Continuing'.format(el))
                return s
        except KeyError:
            print('Warning: {} is not in data dic, hence cannot be loaded'.format(el))
            return s

    def load_get(s, el):
        s.d_load_path(el)
        return s[el]

    def init_data_load_paths(s, data_keys):
        for k in data_keys:
            s[k] = os.path.join(s.path_buffer, k)
        return s

    def to_disk_buffer(s, objects, path=None):
        for fn in objects:
            s.temp_save(path or s.path_buffer, fn)

    def __getstate__(s):
        # Deliberately do not return self.value or self.last_change.
        # We want to have a "blank slate" when we unpickle.
        s.__getattr__ = dict.get
        s.__setattr__ = dict.__setitem__
        s.__delattr__ = dict.__delitem__
        return s

    def __setstate__(s, state):
        # Make self.history = state and last_change and value undefined
        s.__getattr__ = object.__getattribute__
        s.__setattr__ = object.__setattr__
        s.__delattr__ = object.__delattr__
