"""
Base for getting configs from various sources and formats
"""

from collections import ChainMap
from typing import (
    Callable,
    Type,
    Tuple,
    KT,
    VT,
    Any,
    Iterable,
    Protocol,
    Union,
    runtime_checkable,
    Optional,
    MutableMapping,
)
from dataclasses import dataclass
from functools import lru_cache, partial

from config2py.util import always_true, ask_user_for_input, no_default, not_found
from config2py.errors import ConfigNotFound

Exceptions = Tuple[Type[Exception], ...]


@runtime_checkable
class GettableContainer(Protocol):
    """``Containers`` that are "gettable"".

    By "gettable", we mean that we can fetch an element from ``obj`` with brackets:
    ``obj[k]``. That is, ``obj`` has a ``__getitem__`` method.
    A ``Container`` means that ``obj`` has a ``__contains__`` method, i.e. the
    expression ``k in obj`` is valid.

    >>> isinstance(3, GettableContainer)  # 3 is not Gettable (can't do 3[...])
    False

    But ``dict``, ``list``, and ``str`` are GettableContainer:

    >>> isinstance([1, 2, 3], GettableContainer)
    True
    >>> isinstance({'foo': 'bar'}, GettableContainer)
    True
    >>> isinstance('foo', GettableContainer)
    True

    Note that so are their types:

    >>> all(isinstance(c, GettableContainer) for c in (list, dict, str))
    True

    But `set` is not a ``GettableContainer``.

    >>> myset = {1, 2, 3}
    >>> isinstance(myset, GettableContainer)
    False

    This is because a ``set`` is a ``Container``, but it is not gettable:

    >>> 4 in myset  # set is a container
    False
    >>> myset[4]  # ... but not gettable
    Traceback (most recent call last):
    ...
    TypeError: 'set' object is not subscriptable

    """

    def __getitem__(self, k: KT) -> VT:
        pass

    def __contains__(self, k: KT) -> bool:
        pass


Getter = Callable[[KT], VT]
Sources = Iterable[Union[GettableContainer, Getter]]
GetConfigEgress = Callable[[KT, VT], VT]

# # TODO: Refactor into reusable function that can look (and write) in multiple stores
# _open_api_key_env_name = 'OPENAI_API_KEY'
# _api_key = os.environ.get(_open_api_key_env_name, None)
# if _api_key is None:
#     # TODO: Figure out a way to make input response invisible (using * or something)
#     _api_key = getpass.getpass(
#         f"Please set your OpenAI API key and press enter to continue. "
#         f"I will put it in the environment variable {_open_api_key_env_name} "
#     )
# openai.api_key = _api_key


def is_not_none_nor_empty(x):
    if isinstance(x, str):
        return x != ""
    else:
        return x is not None


def get_config(
    key: KT = None,
    sources: Sources = None,
    *,
    default: VT = no_default,
    egress: Optional[GetConfigEgress] = None,
    val_is_valid: Optional[Callable[[VT], bool]] = always_true,
    config_not_found_exceptions: Exceptions = (Exception,),
):
    """Get a config value from a list of sources

    A source can be a function or a ``GettableContainer``.
    (A ``GettableContainer`` is anything that can be indexed with brackets: ``obj[k]``,
    like ``dict``, ``list``, ``str``, etc..).

    Let's take two sources: a ``dict`` and a ``Callable``.

    >>> def func(k):
    ...     if k == 'foo':
    ...         return 'quux'
    ...     elif k == 'green':
    ...         return 'eggs'
    ...     else:
    ...         raise RuntimeError(f"I don't handle that: {k}")
    >>> dict_ = {'foo': 'bar', 'baz': 'qux'}
    >>> sources = [func, dict_]


    See that ``get_config`` go through the sources in the order they were listed,
    and returns the first value it finds (or manages to compute) for the key:

    ``get_config`` finds ``'foo'`` in the very first source (``func``):

    >>> get_config('foo', sources)
    'quux'

    But ``baz`` makes ``func`` raise an error, so it goes to the next source: ``dict_``.
    There, it finds ``'baz'`` and returns its value:

    >>> get_config('baz', sources)
    'qux'

    On the other hand, no one manages to find a config value for ``'no_a_key'``, so
    ``get_config`` raises an error:

    >>> get_config('no_a_key', sources)
    Traceback (most recent call last):
    ...
    config2py.errors.ConfigNotFound: Could not find config for key: no_a_key

    But if you provide a default value, it will return that instead:

    >>> get_config('no_a_key', sources, default='default')
    'default'

    You can also provide a function that will be called on the value before it is
    returned. This is useful if you want to do some post-processing on the value,
    or if you want to make sure that the value is of a certain type:

    This "search the next source if the previous one fails" behavior may not be what
    you want in some situations, since you'd be hiding some errors that you might
    want to be aware of. This is why allow you to specify what exceptions should
    actually be considered as "config not found" exceptions, through the
    ``config_not_found_exceptions`` argument, which defaults to ``Exception``.

    Further, your sources may return a value, but not one that you consider valid:
    For example, a sentinel like ``None``. In this case you may want the search to
    continue. This is what the ``val_is_valid`` argument is for. It is a function
    that takes a value and returns a boolean. If it returns ``False``, the search
    will continue. If it returns ``True``, the search will stop and the value will
    be returned.

    Finally, we have ``egress : Callable[[KT, TT], VT]``.
    This is a function that takes a key and a value, and
    returns a value. It is called after the value has been found, and its return
    value is the one that is returned by ``get_config``. This is useful if you want
    to do some post-processing on the value, or before you return the value, or if you
    want to do some caching.

    >>> config_store = dict()
    >>> def store_before_returning(k, v):
    ...    config_store[k] = v
    ...    return v
    >>> get_config('foo', sources, egress=store_before_returning)
    'quux'
    >>> config_store
    {'foo': 'quux'}

    Note that a source can be a callable or a ``GettableContainer`` (most of the
    time, a ``Mapping`` (e.g. ``dict``)).
    Here, you should be compelled to use the resources of ``dol``
    (https://pypi.org/project/dol/) which will allow you to make ``Mapping``s for all
    sorts of data sources.

    For more info, see: https://github.com/i2mint/config2py/issues/4

    """
    if key is None and sources is not None:
        get_config_ = partial(
            get_config,
            sources=sources,
            default=default,
            egress=egress,
            val_is_valid=val_is_valid,
            config_not_found_exceptions=config_not_found_exceptions,
        )
        get_config_.sources = sources
        return get_config_
    chain_map = sources_chainmap(sources, val_is_valid, config_not_found_exceptions)
    value = chain_map.get(key, not_found)
    if value is not_found:
        if default is no_default:
            raise ConfigNotFound(f"Could not find config for key: {key}")
        else:
            value = default
    if egress is not None:
        return egress(key, value)
    else:
        return value


# TODO: add/enable an __iter__ method if the function's first arg is annotated with
#  an Enum or Literal?
@dataclass
class FuncBasedGettableContainer:
    """A class that wraps a ``Callable[[KT], VT]`` function so it has a (partial)
    Mapping[KT, TT] interface. It is "partial" in the sense that it only implements
    ``__getitem__``, raise a ``KeyError`` when a key can't be computed.
    This is the standard for ``Mapping`` types, which enables us to use the
    ``FuncBasedGettable`` in a ``collections.ChainMap`` to catch the error and move on
    to the next source.

    >>> def getter(k):
    ...     if k == 'foo':
    ...         return 'quux'
    ...     elif k == 'green':
    ...         return 'eggs'
    ...     else:
    ...         raise RuntimeError(f"I don't handle that: {k}")
    >>> gc = FuncBasedGettableContainer(getter)
    >>> gc['foo']
    'quux'
    >>> gc['green']
    'eggs'

    Observe below that though the ``getter`` function raises a ``RuntimeError``, the
    ``FuncBasedGettableContainer`` raises a ``KeyError``, to conform to the
    ``Mapping`` protocol.

    >>> gc['no_a_key']  # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    KeyError: 'There was an exception ... : "I don\'t handle that: no_a_key"'

    Note that by default, ``FuncBasedGettableContainer`` will catch all ``Exception``
    exceptions, but you can specify a different set of exceptions to catch.

    Note as well that you can specify a ``val_is_valid`` function that will be used to
    check the value returned by the ``getter`` function. If the value is not valid, a
    ``KeyError`` will also be raised.
    This is useful, for example, when you have a function that returns a sentinel like
    ``None`` instead of raising an exception, but you want to treat that as a
    ``KeyError``.

    >>> def getter(k):
    ...     if k == 'foo':
    ...         return 'quux'
    ...     elif k == 'green':
    ...         return 'eggs'
    ...     else:
    ...         return None
    >>> gc = FuncBasedGettableContainer(getter, val_is_valid=lambda x: x is not None)
    >>> gc['foo']
    'quux'
    >>> gc['no_a_key']
    Traceback (most recent call last):
    ...
    KeyError: 'Value for key no_a_key is not valid: None'

    """

    getter: Callable[[KT], VT]
    val_is_valid: Callable[[VT], bool] = always_true
    config_not_found_exceptions: Exceptions = (Exception,)
    cache_getter = False

    def __post_init__(self):
        if self.cache_getter:
            # Note: The only purpose of the cache is to avoid calling the getter function
            # twice when doing a ``in`` check before doing a ``[]`` lookup, for instance,
            # in a``collections.ChainMap``.
            # But this caching can lead to unexpected behavior if the getter function
            # has side effects, or if the value it returns changes over time.
            self.getter = lru_cache(maxsize=1)(self.getter)

    def __getitem__(self, k: KT) -> VT:
        try:
            v = self.getter(k)
        except self.config_not_found_exceptions as e:
            raise KeyError(
                f"There was an exception when computing key: {k} with the function "
                f"{self.getter}. The exception was: {e}"
            )
        if not self.val_is_valid(v):
            raise KeyError(f"Value for key {k} is not valid: {v}")
        return v

    # TODO: Is this used to indicate that the getter couldn't find a key.
    def __contains__(self, k):
        try:
            self[k]
        except KeyError:
            return False
        else:
            return True

    def __iter__(self):
        raise TypeError(
            f"Iteration is NOT ACTUALLY implemented for {type(self).__name__}. "
            "The reason we still stuck it in the class is because python will "
            "automatically try to use __getitem__ on 0, 1, ... to iterate over the "
            "object, and we want to avoid that, since this would result in an obscure "
            "error message saying that the getter function can't be called on 0"
        )


def gettable_containers(
    sources: Sources,
    val_is_valid: Callable[[VT], bool] = always_true,
    config_not_found_exceptions: Exceptions = (Exception,),
) -> Iterable[GettableContainer]:
    """Convert an iterable of sources into ``GettableContainers``"""
    for src in sources:
        if isinstance(src, GettableContainer):
            yield src
        elif isinstance(src, Callable):
            yield FuncBasedGettableContainer(
                src,
                val_is_valid=val_is_valid,
                config_not_found_exceptions=config_not_found_exceptions,
            )
        else:
            raise AssertionError(
                f"Source must be a Gettable or a Callable, not {type(src)}"
            )


def sources_chainmap(
    sources: Sources,
    val_is_valid: Callable[[VT], bool] = always_true,
    config_not_found_exceptions: Exceptions = (Exception,),
) -> ChainMap:
    """Create a ``ChainMap`` from a list of sources"""
    sources = gettable_containers(sources, val_is_valid, config_not_found_exceptions)
    return ChainMap(*gettable_containers(sources))


KTSaver = Callable[[KT, VT], Any]
SaveTo = Optional[Union[MutableMapping, KTSaver]]


def is_not_empty(val) -> bool:
    if isinstance(val, str):
        return val != ""
    else:
        return val is not None


def ask_user_for_key(
    key=None,
    *,
    prompt_template="Enter a value for {}: ",
    save_to: SaveTo = None,
    save_condition=is_not_empty,
    user_asker=ask_user_for_input,
    egress: Optional[Callable] = None,
):
    if key is None:
        return partial(
            ask_user_for_key,
            prompt_template=prompt_template,
            save_to=save_to,
            save_condition=save_condition,
            user_asker=user_asker,
            egress=egress,
        )
    val = user_asker(prompt_template.format(key))
    if isinstance(egress, Callable):
        val = egress(key, val)
    if save_to is not None and save_condition(val):
        if hasattr(save_to, "__setitem__"):
            save_to_func = save_to.__setitem__
        save_to_func(key, val)
    return val


def user_gettable(
    save_to: SaveTo = None,
    *,
    prompt_template="Enter a value for {}: ",
    egress: Optional[Callable] = None,
    user_asker=ask_user_for_input,
    val_is_valid: Callable[[VT], bool] = is_not_empty,
    config_not_found_exceptions: Exceptions = (Exception,),
):
    """
    Create a ``GettableContainer`` that asks the user for a value, optionally saving it.

    :param save_to: A ``MutableMapping`` to save the user's response to. If ``None``,
        the user's response is not saved.
    :param prompt_template: A template string to prompt the user with. It should
        contain a placeholder for the key, e.g. ``"Enter a value for {}: "``.
    :param egress: A function to apply to the user's response before returning it.
        This can be used to validate the response, for example.
    :param user_asker: A function that asks the user for input. It should take a
        prompt string and return the user's response.
    :param val_is_valid: A function that takes a value and returns a boolean. If it
        returns ``False``, the user will be asked for a new value.
    :param config_not_found_exceptions: An iterable of exceptions that should be
        considered as "config not found" exceptions. If the user's response raises
        one of these exceptions, the user will be asked for a new value.
    :return: A ``GettableContainer`` that asks the user for a value, optionally saving
        it.

    Example:

    >>> s = user_gettable()
    >>> v = s['SOME_KEY']  # doctest: +SKIP
    'SOME_VAL'

    This will trigger a prompt for the user to enter the value of ``SOME_KEY``.
    When they do (say they entered 'SOME_VAL') it will return that value.

    And if you specify a save_to store (usually a persistent MutableMapping made with
    the ``dol`` package) then it will save the value to that store for future use.

    >>> d = dict(some='store')
    >>> s = user_gettable(save_to=d)
    >>> s['SOME_KEY']  # doctest: +SKIP
    'SOME_VAL'
    >>> d  # doctest: +SKIP
    {'some': 'store', 'SOME_KEY': 'SOME_VAL'}

    """
    getter = ask_user_for_key(
        prompt_template=prompt_template,
        save_to=save_to,
        user_asker=user_asker,
        egress=egress,
    )
    return FuncBasedGettableContainer(
        getter,
        val_is_valid=val_is_valid,
        config_not_found_exceptions=config_not_found_exceptions,
    )
