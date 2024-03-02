"""A registry for managing singleton object warmstarts.

See: https://github.com/braceal/parsl_object_registry/tree/main
"""

from __future__ import annotations

import functools
import inspect
import sys
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import cast
from typing import Generic
from typing import TypeVar

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T = TypeVar('T')
P = ParamSpec('P')


@dataclass
class RegistryInstance(Generic[T]):
    """Store an instance of an object and a shutdown hook."""

    shutdown_callback: Callable[[T], Any] | None = None
    obj: T | None = None
    arg_hash: int = 0

    def shutdown(self) -> None:
        """Shutdown the object."""
        if self.obj is not None:
            if self.shutdown_callback is not None:
                self.shutdown_callback(self.obj)
            self.obj = None
            self.arg_hash = 0


class RegistrySingleton:
    """A registry for managing singleton objects.

    Only one object in the registry can be active at a time.

    Example:
    -------
    Register a class once and then get the singleton instance:

    >>> from pdfwf.registry import registry

    >>> registry.register(MyExpensiveTorchClass)
    >>> my_object = registry.get(MyExpensiveTorchClass, *args, **kwargs)
    """

    _registry: dict[Callable[..., Any], RegistryInstance[Any]]
    _active: Callable[..., Any] | None

    def __new__(cls) -> RegistrySingleton:
        """Create a singleton instance of the registry."""
        if not hasattr(cls, '_instance'):
            cls._instance = super(RegistrySingleton, cls).__new__(cls)  # noqa: UP008
            cls._instance._registry = {}
            cls._instance._active = None
        return cls._instance

    def __contains__(self, cls_fn: Callable[P, T]) -> bool:
        """Check if an object type is in the registry."""
        return cls_fn in self._registry

    def clear(self) -> None:
        """Clear the registry."""
        for instance in self._registry.values():
            instance.shutdown()
        self._registry = {}
        self._active = None

    def register(
        self,
        cls_fn: Callable[P, T],
        shutdown_callback: Callable[[T], Any] | None = None,
    ) -> None:
        """Register an object type with the registry."""
        if cls_fn not in self._registry:
            self._registry[cls_fn] = RegistryInstance(shutdown_callback)

    def get(
        self,
        cls_fn: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Get an object from the registry."""
        # Get the hash of the input arguments to effectively implement an LRU
        # cache with size 1 but with the ability to handle multiple function/
        # class types while only keeping one object active at a time.
        key = hash(
            functools._make_key((cls_fn,) + args, kwargs, typed=False),  # noqa: RUF005
        )

        # Raise an error if the object is not registered
        if cls_fn not in self._registry:
            raise ValueError(f'Object {cls_fn.__name__} not registered.')

        # If the object is already active, then return the previously
        # instantiated object
        if cls_fn == self._active and key == self._registry[cls_fn].arg_hash:
            # There's an internal assertion that if the above two conditions
            # are true, then RegistryInstance.obj is not None. Though since
            # the self._registry dict is unaware of the concrete type of
            # the RegistryInstance generic class, we need to help mypy
            # by casting the type to T.
            return cast(T, self._registry[cls_fn].obj)

        # Shutdown the current active object, if it exists
        if self._active is not None:
            active = self._registry.get(self._active, None)
            if active is not None:
                active.shutdown()

        # Instantiate the new object
        obj = cls_fn(*args, **kwargs)

        # Set the new active object
        self._active = cls_fn
        self._registry[cls_fn].obj = obj
        self._registry[cls_fn].arg_hash = key

        return obj


# Singleton registry
registry = RegistrySingleton()


def _register_fn_decorator(fn: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return registry.get(fn, *args, **kwargs)

    return wrapper


def _register_cls_decorator(cls: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(cls, updated=())
    class SingletonWrapper(cls):  # type: ignore[valid-type,misc]
        def __new__(__cls, *args: P.args, **kwargs: P.kwargs) -> T:  # type: ignore[misc] # noqa: N804
            # Note: We are always calling the registry with the original class.
            # If we called it with this __cls then we would get an infinite
            # recursion loop because __cls is a subclass of cls and the
            # registry would try to instantiate the subclass which would call
            # this method, and the registry again, etc. Instead, we want to
            # instantiate the original class by calling the cls.__init__ in the
            # registry.get method.
            return registry.get(cls, *args, **kwargs)

    return SingletonWrapper


def register(
    shutdown_callback: Callable[[T], Any] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Register a function or class with the registry.

    Parameters
    ----------
    shutdown_callback : Callable[[T], Any], optional
        A function to call when the object is shutdown, by default None

    Example:
    -------
    Register a function once and then get the singleton instance in
    future calls:

    >>> @register(shutdown_callback=clear_torch_cuda_memory_callback)
    >>> def my_expensive_torch_function(*args, **kwargs):
    ...     # Expensive initialization
    ...     return torch_model

    >>> my_object = my_expensive_torch_function(*args, **kwargs)

    Register a class once and then get the singleton instance in future calls:

    >>> @register(shutdown_callback=clear_torch_cuda_memory_callback)
    >>> class MyExpensiveTorchClass:
    ...     def __init__(self, *args, **kwargs) -> None:
    ...         # Expensive initialization
    ...         ...

    >>> my_object = MyExpensiveTorchClass(*args, **kwargs)
    """

    # Note: If a type hint is used, it messes up the intellisense of the
    # decorated function/class. The type should be ClassFn.
    def decorator(cls_fn: Callable[P, T]) -> Callable[P, T]:
        # Register the class/fn immediately when the module is imported
        registry.register(cls_fn, shutdown_callback)

        if inspect.isclass(cls_fn):
            return _register_cls_decorator(cls_fn)
        else:
            return _register_fn_decorator(cls_fn)

    return decorator
