
class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(s):
        # Deliberately do not return self.value or self.last_change.
        # We want to have a "blank slate" when we unpickle.
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
        return s

    def __setstate__(s, state):
        # Make self.history = state and last_change and value undefined
        __getattr__ = object.__getattribute__
        __setattr__ = object.__setattr__
        __delattr__ = object.__delattr__
