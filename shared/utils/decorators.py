import functools
import time

from functools import wraps
from shared.modules.logger import logger


def time_it(func):
    def inner1(*args, **kwargs):
        t0 = time.time()
        return_val = func(*args, **kwargs)
        logger.info("Total Runtime {}: {} seconds".format(func.__name__, round(time.time() - t0, 1)))
        return return_val
    return inner1


def apply_veto_ix(func):
    @wraps(func)
    def inner1(*args, **kwargs):
        stop_ix = func(*args, **kwargs)
        if kwargs['veto_ix'] is not None:
            stop_ix[kwargs['veto_ix']] = False
        return stop_ix
    return inner1


class Alias:
    """
    Alias class that can be used as a decorator for making methods callable
    through other names (or "aliases").
    Note: This decorator must be used inside an @aliased -decorated class.
    For example, if you want to make the method shout() be also callable as
    yell() and scream(), you can use alias like this:

        @alias('yell', 'scream')
        def shout(message):
            # ....
    """

    def __init__(self, *aliases):
        """Constructor."""
        self.aliases = set(aliases)

    def __call__(self, f):
        """
        Method call wrapper. As this decorator has arguments, this method will
        only be called once as a part of the decoration process, receiving only
        one argument: the decorated function ('f'). As a result of this kind of
        decorator, this method must return the callable that will wrap the
        decorated function.
        """
        f._aliases = self.aliases
        return f


def aliased(aliased_class):
    """
    Decorator function that *must* be used in combination with @alias
    decorator. This class will make the magic happen!
    @aliased classes will have their aliased method (via @alias) actually
    aliased.
    This method simply iterates over the member attributes of 'aliased_class'
    seeking for those which have an '_aliases' attribute and then defines new
    members in the class using those aliases as mere pointer functions to the
    original ones.

    Usage:
        @aliased
        class MyClass(object):
            @alias('coolMethod', 'myKinkyMethod')
            def boring_method():
                # ...

        i = MyClass()
        i.coolMethod() # equivalent to i.myKinkyMethod() and i.boring_method()
    """
    original_methods = aliased_class.__dict__.copy()
    for name, method in original_methods.items():
        if hasattr(method, '_aliases'):
            # Add the aliases for 'method', but don't override any
            # previously-defined attribute of 'aliased_class'
            for alias in method._aliases - set(original_methods):
                setattr(aliased_class, alias, method)
    return aliased_class


def property_plus(return_on_error=None):
    def decorate(func):
        # @lru_cache()
        @property
        @wraps(func)
        def inner(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                logger.error(e)
                res = return_on_error
            return res
        return inner
    return decorate


def fluent(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        self = args[0]
        func(*args, **kwargs)
        return self
    return wrapped
