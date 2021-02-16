"""Objects and routines to track debug information."""

import builtins
import traceback
import types
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count


_counter = count(1)


class StackVar:
    """ContextVar that represents a stack."""

    def __init__(self, name):
        """Initialize a StackVar."""
        self.var = ContextVar(name, default=(None, None))
        self.var.set((None, None))

    def push(self, x):
        """Push a new value on the stack."""
        self.var.set((x, self.var.get()))

    def pop(self):
        """Remove the top element of the stack and return it."""
        curr, prev = self.var.get()
        assert prev is not None
        self.var.set(prev)
        return curr

    def top(self):
        """Return the top element of the stack."""
        return self.var.get()[0]


_stack = StackVar("_stack")


def current_info():
    """Return the `DebugInfo` for the current context."""
    return _stack.top()


class DebugInfo:
    """Debug information for an object.

    The `DebugInherit` context manager can be used to automatically
    set certain attributes:

    >>> with debug_inherit(a=1, b=2):
    ...     info = DebugInfo(c=3)
    ...     assert info.a == 1
    ...     assert info.b == 2
    ...     assert info.c == 3

    """

    def __init__(self, obj=None, **kwargs):
        """Construct a DebugInfo object."""
        self.name = None
        self.about = None
        self.relation = None
        self.save_trace = False
        self.trace = None
        self._id = None
        self._obj = None

        top = current_info()
        if top:
            # Only need to look at the top of the stack
            self.__dict__.update(top.__dict__)
        self.__dict__.update(kwargs)

        if obj is not None:
            self._obj = weakref.ref(obj)

        if self.save_trace:
            # We remove the last entry that corresponds to
            # this line in the code.
            self.trace = traceback.extract_stack()[:-1]

    @property
    def id(self):
        if self._id is None:
            self._id = next(_counter)
        return self._id

    @property
    def obj(self):
        """Return the object that this DebugInfo is about."""
        return self._obj and self._obj()

    def set(self, **kwargs):
        self.__dict__.update(kwargs)

    def find(self, prop, skip=builtins.set()):
        """Find a property in self or in self.about."""
        for debug, rel in self.get_chain():
            if hasattr(debug, prop) and rel not in skip:
                return getattr(debug, prop)
        else:
            return None

    def get_chain(self):
        curr = self
        rval = []
        while curr is not None:
            rval.append((curr, curr.relation))
            curr = curr.about
        return rval


@contextmanager
def debug_inherit(**kwargs):
    """Context manager to automatically set attributes on DebugInfo.

    >>> with debug_inherit(a=1, b=2):
    ...     info = DebugInfo(c=3)
    ...     assert info.a == 1
    ...     assert info.b == 2
    ...     assert info.c == 3
    """
    info = DebugInfo(**kwargs)
    _stack.push(info)
    yield
    assert current_info() is info
    _stack.pop()


@contextmanager
def about(parent, relation, **kwargs):
    parent = getattr(parent, "__debuginfo__", parent)
    if not isinstance(parent, DebugInfo):
        raise TypeError("about() takes a DebugInfo or an object with __debuginfo__")
    with debug_inherit(about=parent, relation=relation, **kwargs):
        yield


def attach_debug_info(obj, **kwargs):
    info = DebugInfo(obj, **kwargs)
    obj.__debuginfo__ = info
    return obj


About = about
NamedDebugInfo = DebugInfo
DebugInherit = debug_inherit
