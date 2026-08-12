class recDotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = recDotDict(value)
            self[key] = value

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
